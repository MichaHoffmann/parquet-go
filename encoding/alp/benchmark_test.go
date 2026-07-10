package alp_test

import (
	"testing"

	"github.com/parquet-go/parquet-go/encoding/alp"
)

const benchmarkNumValues = 16 * 1024

func BenchmarkStatefulEncode(b *testing.B) {
	floats := [][]float32{
		makeBenchmarkFloats(100),
		makeBenchmarkFloats(1000),
	}
	doubles := [][]float64{
		makeBenchmarkDoubles(1000),
		makeBenchmarkDoubles(100),
	}

	b.Run("FLOAT/homogeneous", func(b *testing.B) {
		benchmarkStatefulEncodeFloat(b, floats[:1])
	})
	b.Run("FLOAT/alternating", func(b *testing.B) {
		benchmarkStatefulEncodeFloat(b, floats)
	})
	b.Run("DOUBLE/homogeneous", func(b *testing.B) {
		benchmarkStatefulEncodeDouble(b, doubles[:1])
	})
	b.Run("DOUBLE/alternating", func(b *testing.B) {
		benchmarkStatefulEncodeDouble(b, doubles)
	})
}

func benchmarkStatefulEncodeFloat(b *testing.B, pages [][]float32) {
	var base alp.Encoding
	encoder := base.NewStatefulEncoder()
	var dst []byte
	var err error
	for _, page := range pages {
		dst, err = encoder.EncodeFloat(dst[:0], page)
		if err != nil {
			b.Fatal(err)
		}
	}

	rawSize := len(pages[0]) * 4
	b.SetBytes(int64(rawSize))
	b.ReportAllocs()
	var encodedBytes, iterations int64
	page := 0
	for b.Loop() {
		dst, err = encoder.EncodeFloat(dst[:0], pages[page])
		if err != nil {
			b.Fatal(err)
		}
		encodedBytes += int64(len(dst))
		iterations++
		page = (page + 1) % len(pages)
	}
	b.ReportMetric(float64(encodedBytes)/float64(iterations*int64(rawSize)), "ratio")
}

func benchmarkStatefulEncodeDouble(b *testing.B, pages [][]float64) {
	var base alp.Encoding
	encoder := base.NewStatefulEncoder()
	var dst []byte
	var err error
	for _, page := range pages {
		dst, err = encoder.EncodeDouble(dst[:0], page)
		if err != nil {
			b.Fatal(err)
		}
	}

	rawSize := len(pages[0]) * 8
	b.SetBytes(int64(rawSize))
	b.ReportAllocs()
	var encodedBytes, iterations int64
	page := 0
	for b.Loop() {
		dst, err = encoder.EncodeDouble(dst[:0], pages[page])
		if err != nil {
			b.Fatal(err)
		}
		encodedBytes += int64(len(dst))
		iterations++
		page = (page + 1) % len(pages)
	}
	b.ReportMetric(float64(encodedBytes)/float64(iterations*int64(rawSize)), "ratio")
}

func makeBenchmarkFloats(divisor float32) []float32 {
	values := make([]float32, benchmarkNumValues)
	for i := range values {
		values[i] = float32(i%2001-1000) / divisor
	}
	return values
}

func makeBenchmarkDoubles(divisor float64) []float64 {
	values := make([]float64, benchmarkNumValues)
	for i := range values {
		values[i] = float64(i%2001-1000) / divisor
	}
	return values
}
