package alp

import (
	"github.com/parquet-go/parquet-go/encoding"
	"github.com/parquet-go/parquet-go/format"
)

// encodeState carries reusable scratch across pages in one column. Its owning
// ResettableEncoding serializes use and resets it at row-group boundaries.
type encodeState struct {
	floatEnc  floatEncoder
	doubleEnc doubleEncoder
}

// Reset clears page-local presets while retaining scratch buffers.
func (s *encodeState) Reset() {
	s.doubleEnc.presets = nil
	s.floatEnc.presets = nil
}

type statefulEncoder struct {
	encoding.NotSupported
	state encodeState
}

var (
	_ encoding.ResettableEncoding = (*statefulEncoder)(nil)
	_ encoding.ValueCountDecoder  = (*statefulEncoder)(nil)
)

func (e *statefulEncoder) String() string {
	return "ALP"
}

func (e *statefulEncoder) Encoding() format.Encoding {
	return format.ALP
}

func (e *statefulEncoder) Reset() {
	e.state.Reset()
}

func (e *statefulEncoder) EncodeDouble(dst []byte, src []float64) ([]byte, error) {
	var encoding Encoding
	if err := validateEncodeCount(&encoding, len(src)); err != nil {
		return nil, err
	}
	return encoding.encodeDouble(dst, src, &e.state)
}

func (e *statefulEncoder) EncodeFloat(dst []byte, src []float32) ([]byte, error) {
	var encoding Encoding
	if err := validateEncodeCount(&encoding, len(src)); err != nil {
		return nil, err
	}
	return encoding.encodeFloat(dst, src, &e.state)
}

func (e *statefulEncoder) DecodeDouble(dst []float64, src []byte) ([]float64, error) {
	var encoding Encoding
	return encoding.DecodeDouble(dst, src)
}

func (e *statefulEncoder) DecodeFloat(dst []float32, src []byte) ([]float32, error) {
	var encoding Encoding
	return encoding.DecodeFloat(dst, src)
}

func (e *statefulEncoder) DecodeValueCount(src []byte) (int, error) {
	var encoding Encoding
	return encoding.DecodeValueCount(src)
}

func (e *Encoding) encodeDouble(dst []byte, src []float64, state *encodeState) ([]byte, error) {
	if state == nil {
		return e.EncodeDouble(dst, src)
	}
	state.doubleEnc.presets = buildDoublePresetsInto(state.doubleEnc.presetStorage[:], src, defaultVectorSize)
	return encodeDoublePageWith(e, &state.doubleEnc, dst[:0], src)
}

func (e *Encoding) encodeFloat(dst []byte, src []float32, state *encodeState) ([]byte, error) {
	if state == nil {
		return e.EncodeFloat(dst, src)
	}
	state.floatEnc.presets = buildFloatPresetsInto(state.floatEnc.presetStorage[:], src, defaultVectorSize)
	return encodeFloatPageWith(e, &state.floatEnc, dst[:0], src)
}
