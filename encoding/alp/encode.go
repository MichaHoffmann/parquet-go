package alp

import (
	"encoding/binary"
	"math"
	"math/bits"

	"github.com/parquet-go/bitpack"
)

// This file contains the core ALP math and wire serialization. The order of
// floating-point operations is IEEE-754-critical: encode/decode are written as
// single expressions so their rounding matches the reference byte-for-byte.

// evalFloat encodes value under (exponent, factor) and reports whether it cannot
// be reproduced bit-for-bit. The scaling and decode operation order is part of
// the wire compatibility contract and must not be reassociated.
func evalFloat(value float32, exponent, factor int) (encoded int32, exception bool) {
	valueBits := math.Float32bits(value)
	// NaN and ±Inf have all exponent bits set; -0.0 can never be encoded.
	if valueBits&0x7F800000 == 0x7F800000 || valueBits == floatNegativeZeroBits {
		return 0, true
	}
	scaled := value * floatPow10[exponent] * floatPow10Negative[factor]
	// The range comparison also rejects infinities produced by scaling.
	if scaled != scaled || scaled > floatEncodingUpperLimit || scaled < floatEncodingLowerLimit {
		return 0, true
	}
	if scaled >= 0 {
		encoded = int32((scaled + magicFloat) - magicFloat)
	} else {
		encoded = int32((scaled - magicFloat) + magicFloat)
	}
	decoded := float32(encoded) * floatPow10[factor] * floatPow10Negative[exponent]
	return encoded, math.Float32bits(decoded) != valueBits
}

// evalDouble is the double counterpart to evalFloat.
func evalDouble(value float64, exponent, factor int) (encoded int64, exception bool) {
	valueBits := math.Float64bits(value)
	// NaN and ±Inf have all exponent bits set; -0.0 can never be encoded.
	if valueBits&0x7FF0000000000000 == 0x7FF0000000000000 || valueBits == doubleNegativeZeroBits {
		return 0, true
	}
	scaled := value * doublePow10[exponent] * doublePow10Negative[factor]
	if scaled != scaled || scaled > doubleEncodingUpperLimit || scaled < doubleEncodingLowerLimit {
		return 0, true
	}
	if scaled >= 0 {
		encoded = int64((scaled + magicDouble) - magicDouble)
	} else {
		encoded = int64((scaled - magicDouble) + magicDouble)
	}
	decoded := float64(encoded) * doublePow10[factor] * doublePow10Negative[exponent]
	return encoded, math.Float64bits(decoded) != valueBits
}

// bitWidthForUint returns the number of bits needed to represent maxDelta as an
// unsigned value.
func bitWidthForUint(maxDelta uint64) int {
	if maxDelta == 0 {
		return 0
	}
	return 64 - bits.LeadingZeros64(maxDelta)
}

// floatEncoder holds reusable scratch buffers and parameter presets for float
// encoding.
type floatEncoder struct {
	encoded       []uint32
	excPos        []uint16
	presetStorage [maxPresetCombinations][2]int

	// A nil preset set selects a full parameter search for every vector.
	presets [][2]int
}

// buildFloatPresets returns the most frequent parameter pairs from evenly
// spaced sample vectors. It returns nil when every vector should be searched.
func buildFloatPresets(src []float32, vectorSize int) [][2]int {
	var storage [maxPresetCombinations][2]int
	presets := buildFloatPresetsInto(storage[:], src, vectorSize)
	return append([][2]int(nil), presets...)
}

func buildFloatPresetsInto(dst [][2]int, src []float32, vectorSize int) [][2]int {
	numVectors := (len(src) + vectorSize - 1) / vectorSize
	if numVectors <= samplerSampleVectorsPerPage {
		return nil
	}
	n := samplerSampleVectorsPerPage
	var storage [samplerSampleVectorsPerPage]presetCount
	var sampleStorage [samplerValuesPerVector]float32
	counts := storage[:0]
	for i := range n {
		vector := sampleVectorIndex(i, n, numVectors)
		start := vector * vectorSize
		end := min(start+vectorSize, len(src))
		sample := subsampleFloat(sampleStorage[:], src[start:end])
		e, f, _, _, _ := findBestFloatParams(sample)
		counts = countPreset(counts, [2]int{e, f})
	}
	return rankPresets(dst, counts)
}

func subsampleFloat(dst, src []float32) []float32 {
	n := min(len(src), samplerValuesPerVector)
	dst = dst[:n]
	for i := range n {
		dst[i] = src[i*len(src)/n]
	}
	return dst
}

// appendVector encodes a single vector of float values into dst, returning the
// grown slice.
func (s *floatEncoder) appendVector(e *Encoding, dst []byte, offsetTableStart int, values []float32) ([]byte, error) {
	vectorLen := len(values)
	if cap(s.encoded) < vectorLen {
		s.encoded = make([]uint32, vectorLen)
	}
	s.encoded = s.encoded[:vectorLen]
	encoded := s.encoded

	var exponent, factor, numExceptions int
	var minEnc, maxEnc int32
	singletonPreset := useAnalyzeAVX2 && len(s.presets) == 1
	if singletonPreset {
		exponent, factor = s.presets[0][0], s.presets[0][1]
		// The analyzer compacts exceptions into vector-sized scratch.
		if cap(s.excPos) < vectorLen {
			s.excPos = make([]uint16, vectorLen)
		}
		s.excPos = s.excPos[:vectorLen]
		numExceptions, minEnc, maxEnc = analyzeFloatEncoded(values, exponent, factor, encoded, s.excPos)
		if numExceptions == vectorLen {
			minEnc, maxEnc = 0, 0
		}
	} else if s.presets != nil {
		exponent, factor, numExceptions, minEnc, maxEnc = findBestFloatParamsWithPresets(values, 0, vectorLen, s.presets)
	} else {
		exponent, factor, numExceptions, minEnc, maxEnc = findBestFloatParams(values)
	}

	if singletonPreset {
		s.excPos = s.excPos[:numExceptions]
	} else {
		if cap(s.excPos) < numExceptions {
			s.excPos = make([]uint16, numExceptions)
		}
		s.excPos = s.excPos[:numExceptions]
	}

	// The search supplies the frame of reference and range. The encode paths
	// initially zero exception lanes; the post-pass below replaces them with the
	// first non-exception delta, or leaves zero when every lane is an exception.
	minValue := minEnc
	umin := uint32(minEnc)
	maxDelta := uint32(maxEnc) - umin

	if singletonPreset {
		for i := range encoded {
			encoded[i] -= umin
		}
	} else if useEncodeAVX2 {
		// The SIMD kernel produces final deltas and compact exception positions.
		encodeFloatDeltas(values, exponent, factor, minEnc, encoded, s.excPos)
	} else {
		// Pure-scalar path: compute inline.
		exceptionIndex := 0
		if numExceptions == 0 {
			for i := range vectorLen {
				enc, _ := evalFloat(values[i], exponent, factor)
				encoded[i] = uint32(enc) - umin
			}
		} else {
			for i := range vectorLen {
				enc, exc := evalFloat(values[i], exponent, factor)
				if exc {
					s.excPos[exceptionIndex] = uint16(i)
					exceptionIndex++
					encoded[i] = 0
				} else {
					encoded[i] = uint32(enc) - umin
				}
			}
		}
	}
	normalizeFloatExceptionPlaceholders(encoded, s.excPos)
	bitWidth := bitWidthForUint(uint64(maxDelta))
	packedBytes := bitpack.ByteCount(uint(vectorLen) * uint(bitWidth))
	vectorBytes := alpInfoSize + floatForInfoSize + packedBytes + numExceptions*(2+4)
	if err := validateVectorOutputSize(e, dst, offsetTableStart, vectorBytes); err != nil {
		return nil, err
	}

	// AlpInfo: exponent(1)+factor(1)+num_exceptions(2)
	dst = append(dst, byte(exponent), byte(factor))
	dst = binary.LittleEndian.AppendUint16(dst, uint16(numExceptions))
	// ForInfo: frame_of_reference(4)+bit_width(1)
	dst = binary.LittleEndian.AppendUint32(dst, uint32(minValue))
	dst = append(dst, byte(bitWidth))

	if bitWidth > 0 {
		packedStart := len(dst)
		dst = extendBytes(dst, packedBytes)
		bitpack.Pack(dst[packedStart:], encoded, uint(bitWidth))
	}

	if numExceptions > 0 {
		for _, pos := range s.excPos {
			dst = binary.LittleEndian.AppendUint16(dst, pos)
		}
		for _, pos := range s.excPos {
			dst = binary.LittleEndian.AppendUint32(dst, math.Float32bits(values[pos]))
		}
	}
	return dst, nil
}

func normalizeFloatExceptionPlaceholders(encoded []uint32, exceptionPositions []uint16) {
	var placeholder uint32
	if len(exceptionPositions) < len(encoded) {
		exception := 0
		for i := range encoded {
			if exception < len(exceptionPositions) && int(exceptionPositions[exception]) == i {
				exception++
				continue
			}
			placeholder = encoded[i]
			break
		}
	}
	for _, pos := range exceptionPositions {
		encoded[pos] = placeholder
	}
}

func validateVectorOutputSize(e *Encoding, dst []byte, offsetTableStart, vectorBytes int) error {
	const maxOffset = uint64(math.MaxUint32)
	outputSize := uint64(len(dst)-offsetTableStart) + uint64(vectorBytes)
	if outputSize > maxOffset {
		return wrapf(e, "ALP encoded page size %d exceeds maximum offset %d", outputSize, maxOffset)
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(len(dst))+uint64(vectorBytes) > maxInt {
		return wrapf(e, "ALP encoded output size overflows int")
	}
	return nil
}

// extendBytes grows b by n bytes without clearing reused capacity. Callers must
// overwrite every added byte.
func extendBytes(b []byte, n int) []byte {
	need := len(b) + n
	if cap(b) < need {
		nb := make([]byte, need)
		copy(nb, b)
		return nb
	}
	return b[:need]
}

// encodeFloatPage encodes src as a complete ALP page appended to dst.
func encodeFloatPage(e *Encoding, dst []byte, src []float32) ([]byte, error) {
	return encodeFloatPageWithPresets(e, dst, src, buildFloatPresets(src, defaultVectorSize))
}

// encodeFloatPageWithPresets encodes src with the supplied parameter presets.
// A nil set full-searches every vector.
func encodeFloatPageWithPresets(e *Encoding, dst []byte, src []float32, presets [][2]int) ([]byte, error) {
	return encodeFloatPageWith(e, &floatEncoder{presets: presets}, dst, src)
}

// encodeFloatPageWith encodes src with reusable scratch and backpatches vector
// offsets as each vector is appended.
func encodeFloatPageWith(e *Encoding, enc *floatEncoder, dst []byte, src []float32) ([]byte, error) {
	dst = appendHeader(dst, defaultVectorSizeLog, len(src))
	if len(src) == 0 {
		return dst, nil
	}

	numVectors := (len(src) + defaultVectorSize - 1) / defaultVectorSize
	offsetTableSize := numVectors * 4

	offsetTableStart := len(dst)
	dst = extendBytes(dst, offsetTableSize)
	for v := range numVectors {
		binary.LittleEndian.PutUint32(dst[offsetTableStart+v*4:], uint32(len(dst)-offsetTableStart))
		start := v * defaultVectorSize
		end := min(start+defaultVectorSize, len(src))
		var err error
		dst, err = enc.appendVector(e, dst, offsetTableStart, src[start:end])
		if err != nil {
			return nil, err
		}
	}
	return dst, nil
}

// doubleEncoder holds reusable encode scratch and parameter presets.
type doubleEncoder struct {
	encoded       []uint64
	excPos        []uint16
	presetStorage [maxPresetCombinations][2]int

	presets [][2]int
}

// buildDoublePresets is the double counterpart to buildFloatPresets.
func buildDoublePresets(src []float64, vectorSize int) [][2]int {
	var storage [maxPresetCombinations][2]int
	presets := buildDoublePresetsInto(storage[:], src, vectorSize)
	return append([][2]int(nil), presets...)
}

func buildDoublePresetsInto(dst [][2]int, src []float64, vectorSize int) [][2]int {
	numVectors := (len(src) + vectorSize - 1) / vectorSize
	if numVectors <= samplerSampleVectorsPerPage {
		return nil
	}
	n := samplerSampleVectorsPerPage
	var storage [samplerSampleVectorsPerPage]presetCount
	var sampleStorage [samplerValuesPerVector]float64
	counts := storage[:0]
	for i := range n {
		vector := sampleVectorIndex(i, n, numVectors)
		start := vector * vectorSize
		end := min(start+vectorSize, len(src))
		sample := subsampleDouble(sampleStorage[:], src[start:end])
		e, f, _, _, _ := findBestDoubleParams(sample)
		counts = countPreset(counts, [2]int{e, f})
	}
	return rankPresets(dst, counts)
}

func subsampleDouble(dst, src []float64) []float64 {
	n := min(len(src), samplerValuesPerVector)
	dst = dst[:n]
	for i := range n {
		dst[i] = src[i*len(src)/n]
	}
	return dst
}

// appendVector encodes a single vector of double values into dst.
func (s *doubleEncoder) appendVector(e *Encoding, dst []byte, offsetTableStart int, values []float64) ([]byte, error) {
	vectorLen := len(values)

	var exponent, factor, numExceptions int
	var minEnc, maxEnc int64
	if s.presets != nil {
		exponent, factor, numExceptions, minEnc, maxEnc = findBestDoubleParamsWithPresets(values, 0, vectorLen, s.presets)
	} else {
		exponent, factor, numExceptions, minEnc, maxEnc = findBestDoubleParams(values)
	}

	if cap(s.encoded) < vectorLen {
		s.encoded = make([]uint64, vectorLen)
	}
	s.encoded = s.encoded[:vectorLen]
	encoded := s.encoded
	if cap(s.excPos) < numExceptions {
		s.excPos = make([]uint16, numExceptions)
	}
	s.excPos = s.excPos[:numExceptions]
	exceptionIndex := 0

	// The search supplies the frame of reference and encoded range.
	minValue := minEnc
	umin := uint64(minEnc)
	maxDelta := uint64(maxEnc) - umin

	if numExceptions == 0 {
		for i := range vectorLen {
			enc, _ := evalDouble(values[i], exponent, factor)
			encoded[i] = uint64(enc) - umin
		}
	} else {
		for i := range vectorLen {
			enc, exc := evalDouble(values[i], exponent, factor)
			if exc {
				s.excPos[exceptionIndex] = uint16(i)
				exceptionIndex++
				encoded[i] = 0
			} else {
				encoded[i] = uint64(enc) - umin
			}
		}
	}
	normalizeDoubleExceptionPlaceholders(encoded, s.excPos)
	bitWidth := bitWidthForUint(maxDelta)
	packedBytes := bitpack.ByteCount(uint(vectorLen) * uint(bitWidth))
	vectorBytes := alpInfoSize + doubleForInfoSize + packedBytes + numExceptions*(2+8)
	if err := validateVectorOutputSize(e, dst, offsetTableStart, vectorBytes); err != nil {
		return nil, err
	}

	// AlpInfo: exponent(1)+factor(1)+num_exceptions(2)
	dst = append(dst, byte(exponent), byte(factor))
	dst = binary.LittleEndian.AppendUint16(dst, uint16(numExceptions))
	// ForInfo: frame_of_reference(8)+bit_width(1)
	dst = binary.LittleEndian.AppendUint64(dst, uint64(minValue))
	dst = append(dst, byte(bitWidth))

	if bitWidth > 0 {
		packedStart := len(dst)
		dst = extendBytes(dst, packedBytes)
		bitpack.Pack(dst[packedStart:], encoded, uint(bitWidth))
	}

	if numExceptions > 0 {
		for _, pos := range s.excPos {
			dst = binary.LittleEndian.AppendUint16(dst, pos)
		}
		for _, pos := range s.excPos {
			dst = binary.LittleEndian.AppendUint64(dst, math.Float64bits(values[pos]))
		}
	}
	return dst, nil
}

func normalizeDoubleExceptionPlaceholders(encoded []uint64, exceptionPositions []uint16) {
	var placeholder uint64
	if len(exceptionPositions) < len(encoded) {
		exception := 0
		for i := range encoded {
			if exception < len(exceptionPositions) && int(exceptionPositions[exception]) == i {
				exception++
				continue
			}
			placeholder = encoded[i]
			break
		}
	}
	for _, pos := range exceptionPositions {
		encoded[pos] = placeholder
	}
}

// encodeDoublePage encodes src as a complete ALP page appended to dst.
func encodeDoublePage(e *Encoding, dst []byte, src []float64) ([]byte, error) {
	return encodeDoublePageWithPresets(e, dst, src, buildDoublePresets(src, defaultVectorSize))
}

// encodeDoublePageWithPresets is the double counterpart to
// encodeFloatPageWithPresets.
func encodeDoublePageWithPresets(e *Encoding, dst []byte, src []float64, presets [][2]int) ([]byte, error) {
	return encodeDoublePageWith(e, &doubleEncoder{presets: presets}, dst, src)
}

// encodeDoublePageWith is the double counterpart to encodeFloatPageWith.
func encodeDoublePageWith(e *Encoding, enc *doubleEncoder, dst []byte, src []float64) ([]byte, error) {
	dst = appendHeader(dst, defaultVectorSizeLog, len(src))
	if len(src) == 0 {
		return dst, nil
	}

	numVectors := (len(src) + defaultVectorSize - 1) / defaultVectorSize
	offsetTableSize := numVectors * 4

	offsetTableStart := len(dst)
	dst = extendBytes(dst, offsetTableSize)
	for v := range numVectors {
		binary.LittleEndian.PutUint32(dst[offsetTableStart+v*4:], uint32(len(dst)-offsetTableStart))
		start := v * defaultVectorSize
		end := min(start+defaultVectorSize, len(src))
		var err error
		dst, err = enc.appendVector(e, dst, offsetTableStart, src[start:end])
		if err != nil {
			return nil, err
		}
	}
	return dst, nil
}

// appendHeader appends the 7-byte ALP page header.
func appendHeader(dst []byte, logVectorSize, numElements int) []byte {
	dst = append(dst, byte(alpCompressionMode), byte(alpIntegerEncodingFOR), byte(logVectorSize))
	dst = binary.LittleEndian.AppendUint32(dst, uint32(numElements))
	return dst
}
