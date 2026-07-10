// Package alp implements the experimental ALP (Adaptive Lossless floating-Point)
// encoding for Parquet FLOAT and DOUBLE columns.
//
// ALP converts floating-point values to integers using decimal scaling, then
// applies Frame of Reference encoding and bit-packing. Values that cannot be
// losslessly converted are stored verbatim as exceptions, so the encoding is
// exact-bit lossless for every input (including NaN payloads, ±Inf and -0.0).
//
// This encoding is opt-in. Its provisional wire format is not yet part of the
// ratified parquet-format specification. It uses experimental Parquet encoding
// ID 10.
//
// Based on the paper: "ALP: Adaptive Lossless floating-Point Compression"
// (SIGMOD 2024), https://dl.acm.org/doi/10.1145/3626717
package alp

import (
	"github.com/parquet-go/parquet-go/encoding"
	"github.com/parquet-go/parquet-go/format"
)

// Encoding implements the experimental ALP encoding for FLOAT and DOUBLE
// columns.
//
// Only EncodeFloat/DecodeFloat and EncodeDouble/DecodeDouble are supported; all
// other value types fall through to encoding.NotSupported.
type Encoding struct {
	encoding.NotSupported
}

var (
	_ encoding.Encoding          = (*Encoding)(nil)
	_ encoding.ValueCountDecoder = (*Encoding)(nil)
	_ interface {
		NewStatefulEncoder() encoding.ResettableEncoding
	} = (*Encoding)(nil)
)

// NewStatefulEncoder returns an encoder that rebuilds sampled presets per page
// while carrying scratch buffers across pages. The returned mutable encoder
// requires exclusive ownership by one column writer; its Encode, Decode, and
// Reset methods must not overlap, and Reset must be called between row groups.
func (e *Encoding) NewStatefulEncoder() encoding.ResettableEncoding {
	return new(statefulEncoder)
}

// String returns the human-readable name of the encoding.
func (e *Encoding) String() string {
	return "ALP"
}

// Encoding returns the parquet format code for ALP.
func (e *Encoding) Encoding() format.Encoding {
	return format.ALP
}

// EncodeFloat encodes src as an ALP page starting at dst[:0], reusing capacity.
func (e *Encoding) EncodeFloat(dst []byte, src []float32) ([]byte, error) {
	if err := validateEncodeCount(e, len(src)); err != nil {
		return nil, err
	}
	return encodeFloatPage(e, dst[:0], src)
}

// EncodeDouble encodes src as an ALP page starting at dst[:0], reusing capacity.
func (e *Encoding) EncodeDouble(dst []byte, src []float64) ([]byte, error) {
	if err := validateEncodeCount(e, len(src)); err != nil {
		return nil, err
	}
	return encodeDoublePage(e, dst[:0], src)
}

func validateEncodeCount(e *Encoding, count int) error {
	const maxElements = int64(1<<31 - 1)
	if int64(count) > maxElements {
		return wrapf(e, "ALP element count %d exceeds maximum %d", count, maxElements)
	}
	return nil
}

// DecodeFloat decodes an ALP float page from src into dst.
func (e *Encoding) DecodeFloat(dst []float32, src []byte) ([]float32, error) {
	return decodeFloatPage(e, dst, src)
}

// DecodeDouble decodes an ALP double page from src into dst.
func (e *Encoding) DecodeDouble(dst []float64, src []byte) ([]float64, error) {
	return decodeDoublePage(e, dst, src)
}

// DecodeValueCount returns the element count declared by an ALP payload.
func (e *Encoding) DecodeValueCount(src []byte) (int, error) {
	_, count, err := decodeHeaderFields(e, src, "ALP")
	return count, err
}

// wrapf builds an error wrapped with the encoding identity.
func wrapf(e *Encoding, msg string, args ...any) error {
	return encoding.Errorf(e, msg, args...)
}
