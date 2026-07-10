package alp

import (
	"encoding/binary"
	"math"

	"github.com/parquet-go/bitpack"
	"github.com/parquet-go/parquet-go/encoding"
)

func decodeFloat(encoded int32, exponent, factor int) float32 {
	return float32(encoded) * floatPow10[factor] * floatPow10Negative[exponent]
}

func decodeDouble(encoded int64, exponent, factor int) float64 {
	return float64(encoded) * doublePow10[factor] * doublePow10Negative[exponent]
}

type alpHeader struct {
	vectorSize      int
	elementCount    int
	numVectors      int
	offsets         []byte
	body            []byte
	offsetTableSize int
}

func parseHeader(e *Encoding, src []byte, typ string, forInfoSize int) (alpHeader, error) {
	var h alpHeader
	logVectorSize, elementCount, err := decodeHeaderFields(e, src, typ)
	if err != nil {
		return h, err
	}
	h = alpHeader{
		vectorSize:   1 << logVectorSize,
		elementCount: elementCount,
	}
	if h.elementCount == 0 {
		if len(src) != alpHeaderSize {
			return h, wrapf(e, "zero-element ALP page has %d trailing bytes", len(src)-alpHeaderSize)
		}
		return h, nil
	}
	h.numVectors = h.elementCount / h.vectorSize
	if h.elementCount%h.vectorSize != 0 {
		h.numVectors++
	}
	rest := src[alpHeaderSize:]
	if h.numVectors > len(rest)/4 {
		return h, encoding.ErrDecodeInvalidInputSize(e, typ, len(src))
	}
	h.offsetTableSize = h.numVectors * 4
	h.offsets = rest[:h.offsetTableSize]
	h.body = rest[h.offsetTableSize:]

	minVectorSize := alpInfoSize + forInfoSize
	if h.numVectors > len(h.body)/minVectorSize {
		return h, wrapf(e, "ALP body too small: have %d bytes, need at least %d for %d vectors", len(h.body), int64(h.numVectors)*int64(minVectorSize), h.numVectors)
	}
	pageEnd := uint64(len(rest))
	previous := uint32(0)
	for v, pos := 0, 0; v < h.numVectors; v, pos = v+1, pos+4 {
		offset := binary.LittleEndian.Uint32(h.offsets[pos:])
		if uint64(offset) < uint64(h.offsetTableSize) || uint64(offset) > pageEnd {
			return h, wrapf(e, "ALP vector %d offset %d out of range", v, offset)
		}
		if v == 0 && uint64(offset) != uint64(h.offsetTableSize) {
			return h, wrapf(e, "invalid first ALP vector offset %d, want %d", offset, h.offsetTableSize)
		}
		if v > 0 && offset <= previous {
			return h, wrapf(e, "ALP vector offsets not strictly increasing at vector %d", v)
		}
		previous = offset
	}
	return h, nil
}

func decodeHeaderFields(e *Encoding, src []byte, typ string) (logVectorSize, elementCount int, err error) {
	if len(src) < alpHeaderSize {
		return 0, 0, encoding.ErrDecodeInvalidInputSize(e, typ, len(src))
	}
	compressionMode := int(src[0])
	integerEncoding := int(src[1])
	logVectorSize = int(src[2])
	numElements := binary.LittleEndian.Uint32(src[3:7])
	if compressionMode != alpCompressionMode {
		return 0, 0, wrapf(e, "unsupported ALP compression mode: %d", compressionMode)
	}
	if integerEncoding != alpIntegerEncodingFOR {
		return 0, 0, wrapf(e, "unsupported ALP integer encoding: %d", integerEncoding)
	}
	if logVectorSize < minLogVectorSize || logVectorSize > maxLogVectorSize {
		return 0, 0, wrapf(e, "invalid ALP log vector size: %d, must be between %d and %d", logVectorSize, minLogVectorSize, maxLogVectorSize)
	}
	if numElements > math.MaxInt32 {
		return 0, 0, wrapf(e, "invalid ALP element count: %d", numElements)
	}
	return logVectorSize, int(numElements), nil
}

func (h *alpHeader) vectorLength(idx int) int {
	if idx < h.numVectors-1 {
		return h.vectorSize
	}
	last := h.elementCount % h.vectorSize
	if last == 0 {
		return h.vectorSize
	}
	return last
}

func (h *alpHeader) vectorRegion(idx int) []byte {
	start := int(binary.LittleEndian.Uint32(h.offsets[idx*4:])) - h.offsetTableSize
	end := len(h.body)
	if idx+1 < h.numVectors {
		end = int(binary.LittleEndian.Uint32(h.offsets[(idx+1)*4:])) - h.offsetTableSize
	}
	return h.body[start:end]
}

func growFloat32(dst []float32, n int) []float32 {
	if cap(dst)-len(dst) >= n {
		return dst[:len(dst)+n]
	}
	return append(dst, make([]float32, n)...)
}

func growFloat64(dst []float64, n int) []float64 {
	if cap(dst)-len(dst) >= n {
		return dst[:len(dst)+n]
	}
	return append(dst, make([]float64, n)...)
}

type floatVector struct {
	exponent         int
	factor           int
	frameOfReference int32
	bitWidth         int
	packed           []byte
	exceptionPos     []byte
	exceptionValues  []byte
}

func parseFloatVector(e *Encoding, vector []byte, vectorLen, vectorIndex int) (floatVector, error) {
	var meta floatVector
	if len(vector) < alpInfoSize {
		return meta, wrapf(e, "ALP vector %d truncated ALP info", vectorIndex)
	}
	meta.exponent = int(vector[0])
	meta.factor = int(vector[1])
	numExceptions := int(binary.LittleEndian.Uint16(vector[2:]))
	if meta.exponent > floatMaxExponent {
		return meta, wrapf(e, "invalid ALP float exponent %d in vector %d, max is %d", meta.exponent, vectorIndex, floatMaxExponent)
	}
	if meta.factor > meta.exponent {
		return meta, wrapf(e, "invalid ALP float factor %d > exponent %d in vector %d", meta.factor, meta.exponent, vectorIndex)
	}
	if numExceptions > vectorLen {
		return meta, wrapf(e, "invalid ALP numExceptions %d > vectorLen %d in vector %d", numExceptions, vectorLen, vectorIndex)
	}

	pos := alpInfoSize
	if len(vector)-pos < floatForInfoSize {
		return meta, wrapf(e, "ALP vector %d truncated FOR info", vectorIndex)
	}
	meta.frameOfReference = int32(binary.LittleEndian.Uint32(vector[pos:]))
	meta.bitWidth = int(vector[pos+4])
	if meta.bitWidth > 32 {
		return meta, wrapf(e, "invalid ALP float bitWidth %d > 32 in vector %d", meta.bitWidth, vectorIndex)
	}
	pos += floatForInfoSize

	packedBytes := (uint64(vectorLen)*uint64(meta.bitWidth) + 7) / 8
	if packedBytes > uint64(len(vector)-pos) {
		return meta, wrapf(e, "ALP vector %d truncated packed data", vectorIndex)
	}
	meta.packed = vector[pos : pos+int(packedBytes)]
	pos += int(packedBytes)

	exceptionBytes := numExceptions * (2 + 4)
	if len(vector)-pos < exceptionBytes {
		return meta, wrapf(e, "ALP vector %d truncated exceptions", vectorIndex)
	}
	if len(vector)-pos > exceptionBytes {
		return meta, wrapf(e, "ALP vector %d has %d trailing bytes", vectorIndex, len(vector)-pos-exceptionBytes)
	}
	meta.exceptionPos = vector[pos : pos+numExceptions*2]
	meta.exceptionValues = vector[pos+numExceptions*2:]
	for ex := range numExceptions {
		p := int(binary.LittleEndian.Uint16(meta.exceptionPos[ex*2:]))
		if p >= vectorLen {
			return meta, wrapf(e, "ALP exception position %d out of bounds for vectorLen %d", p, vectorLen)
		}
	}
	return meta, nil
}

type doubleVector struct {
	exponent         int
	factor           int
	frameOfReference int64
	bitWidth         int
	packed           []byte
	exceptionPos     []byte
	exceptionValues  []byte
}

func parseDoubleVector(e *Encoding, vector []byte, vectorLen, vectorIndex int) (doubleVector, error) {
	var meta doubleVector
	if len(vector) < alpInfoSize {
		return meta, wrapf(e, "ALP vector %d truncated ALP info", vectorIndex)
	}
	meta.exponent = int(vector[0])
	meta.factor = int(vector[1])
	numExceptions := int(binary.LittleEndian.Uint16(vector[2:]))
	if meta.exponent > doubleMaxExponent {
		return meta, wrapf(e, "invalid ALP double exponent %d in vector %d, max is %d", meta.exponent, vectorIndex, doubleMaxExponent)
	}
	if meta.factor > meta.exponent {
		return meta, wrapf(e, "invalid ALP double factor %d > exponent %d in vector %d", meta.factor, meta.exponent, vectorIndex)
	}
	if numExceptions > vectorLen {
		return meta, wrapf(e, "invalid ALP numExceptions %d > vectorLen %d in vector %d", numExceptions, vectorLen, vectorIndex)
	}

	pos := alpInfoSize
	if len(vector)-pos < doubleForInfoSize {
		return meta, wrapf(e, "ALP vector %d truncated FOR info", vectorIndex)
	}
	meta.frameOfReference = int64(binary.LittleEndian.Uint64(vector[pos:]))
	meta.bitWidth = int(vector[pos+8])
	if meta.bitWidth > 64 {
		return meta, wrapf(e, "invalid ALP double bitWidth %d > 64 in vector %d", meta.bitWidth, vectorIndex)
	}
	pos += doubleForInfoSize

	packedBytes := (uint64(vectorLen)*uint64(meta.bitWidth) + 7) / 8
	if packedBytes > uint64(len(vector)-pos) {
		return meta, wrapf(e, "ALP vector %d truncated packed data", vectorIndex)
	}
	meta.packed = vector[pos : pos+int(packedBytes)]
	pos += int(packedBytes)

	exceptionBytes := numExceptions * (2 + 8)
	if len(vector)-pos < exceptionBytes {
		return meta, wrapf(e, "ALP vector %d truncated exceptions", vectorIndex)
	}
	if len(vector)-pos > exceptionBytes {
		return meta, wrapf(e, "ALP vector %d has %d trailing bytes", vectorIndex, len(vector)-pos-exceptionBytes)
	}
	meta.exceptionPos = vector[pos : pos+numExceptions*2]
	meta.exceptionValues = vector[pos+numExceptions*2:]
	for ex := range numExceptions {
		p := int(binary.LittleEndian.Uint16(meta.exceptionPos[ex*2:]))
		if p >= vectorLen {
			return meta, wrapf(e, "ALP exception position %d out of bounds for vectorLen %d", p, vectorLen)
		}
	}
	return meta, nil
}

func decodeFloatPage(e *Encoding, dst []float32, src []byte) ([]float32, error) {
	h, err := parseHeader(e, src, "FLOAT", floatForInfoSize)
	if err != nil {
		return dst[:0], err
	}
	if uint64(h.elementCount)*4 > uint64(^uintptr(0)) {
		return dst[:0], wrapf(e, "decoded FLOAT output for %d elements exceeds uintptr address space", h.elementCount)
	}
	dst = dst[:0]
	if h.elementCount == 0 {
		return dst, nil
	}
	for v := 0; v < h.numVectors; v++ {
		if _, err := parseFloatVector(e, h.vectorRegion(v), h.vectorLength(v), v); err != nil {
			return dst, err
		}
	}

	deltas := make([]int32, h.vectorSize)
	var scratch []byte
	for v := 0; v < h.numVectors; v++ {
		vectorLen := h.vectorLength(v)
		meta, _ := parseFloatVector(e, h.vectorRegion(v), vectorLen, v)
		if meta.bitWidth > 0 {
			packedBytes := len(meta.packed)
			need := packedBytes + bitpack.PaddingInt32
			if cap(scratch) < need {
				scratch = make([]byte, need)
			} else {
				scratch = scratch[:need]
			}
			copy(scratch, meta.packed)
			for i := packedBytes; i < need; i++ {
				scratch[i] = 0
			}
			bitpack.Unpack(deltas[:vectorLen], scratch, uint(meta.bitWidth))
		} else {
			for i := range vectorLen {
				deltas[i] = 0
			}
		}

		base := len(dst)
		dst = growFloat32(dst, vectorLen)
		for i := range vectorLen {
			encoded := deltas[i] + meta.frameOfReference
			dst[base+i] = decodeFloat(encoded, meta.exponent, meta.factor)
		}
		for ex := 0; ex < len(meta.exceptionPos)/2; ex++ {
			p := int(binary.LittleEndian.Uint16(meta.exceptionPos[ex*2:]))
			bitsv := binary.LittleEndian.Uint32(meta.exceptionValues[ex*4:])
			dst[base+p] = math.Float32frombits(bitsv)
		}
	}
	return dst, nil
}

func decodeDoublePage(e *Encoding, dst []float64, src []byte) ([]float64, error) {
	h, err := parseHeader(e, src, "DOUBLE", doubleForInfoSize)
	if err != nil {
		return dst[:0], err
	}
	if uint64(h.elementCount)*8 > uint64(^uintptr(0)) {
		return dst[:0], wrapf(e, "decoded DOUBLE output for %d elements exceeds uintptr address space", h.elementCount)
	}
	dst = dst[:0]
	if h.elementCount == 0 {
		return dst, nil
	}
	for v := 0; v < h.numVectors; v++ {
		if _, err := parseDoubleVector(e, h.vectorRegion(v), h.vectorLength(v), v); err != nil {
			return dst, err
		}
	}

	deltas := make([]int64, h.vectorSize)
	var scratch []byte
	for v := 0; v < h.numVectors; v++ {
		vectorLen := h.vectorLength(v)
		meta, _ := parseDoubleVector(e, h.vectorRegion(v), vectorLen, v)
		if meta.bitWidth > 0 {
			packedBytes := len(meta.packed)
			need := packedBytes + bitpack.PaddingInt64
			if cap(scratch) < need {
				scratch = make([]byte, need)
			} else {
				scratch = scratch[:need]
			}
			copy(scratch, meta.packed)
			for i := packedBytes; i < need; i++ {
				scratch[i] = 0
			}
			bitpack.Unpack(deltas[:vectorLen], scratch, uint(meta.bitWidth))
		} else {
			for i := range vectorLen {
				deltas[i] = 0
			}
		}

		base := len(dst)
		dst = growFloat64(dst, vectorLen)
		for i := range vectorLen {
			encoded := deltas[i] + meta.frameOfReference
			dst[base+i] = decodeDouble(encoded, meta.exponent, meta.factor)
		}
		for ex := 0; ex < len(meta.exceptionPos)/2; ex++ {
			p := int(binary.LittleEndian.Uint16(meta.exceptionPos[ex*2:]))
			bitsv := binary.LittleEndian.Uint64(meta.exceptionValues[ex*8:])
			dst[base+p] = math.Float64frombits(bitsv)
		}
	}
	return dst, nil
}
