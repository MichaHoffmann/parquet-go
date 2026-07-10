package alp

import "testing"

func TestSubsampleVector(t *testing.T) {
	floats := make([]float32, defaultVectorSize)
	doubles := make([]float64, defaultVectorSize)
	for i := range floats {
		floats[i] = float32(i)
		doubles[i] = float64(i)
	}

	floatSample := subsampleFloat(make([]float32, samplerValuesPerVector), floats)
	doubleSample := subsampleDouble(make([]float64, samplerValuesPerVector), doubles)
	if len(floatSample) != samplerValuesPerVector || len(doubleSample) != samplerValuesPerVector {
		t.Fatalf("sample lengths = (%d, %d), want %d", len(floatSample), len(doubleSample), samplerValuesPerVector)
	}
	for i := range samplerValuesPerVector {
		want := i * defaultVectorSize / samplerValuesPerVector
		if floatSample[i] != float32(want) || doubleSample[i] != float64(want) {
			t.Fatalf("sample %d = (%v, %v), want %d", i, floatSample[i], doubleSample[i], want)
		}
	}

	shortFloats := floats[:17]
	shortDoubles := doubles[:17]
	floatSample = subsampleFloat(floatSample, shortFloats)
	doubleSample = subsampleDouble(doubleSample, shortDoubles)
	if len(floatSample) != len(shortFloats) || len(doubleSample) != len(shortDoubles) {
		t.Fatalf("short sample lengths = (%d, %d), want %d", len(floatSample), len(doubleSample), len(shortFloats))
	}
	for i := range shortFloats {
		if floatSample[i] != shortFloats[i] || doubleSample[i] != shortDoubles[i] {
			t.Fatalf("short sample %d changed", i)
		}
	}
}
