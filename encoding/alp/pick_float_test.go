package alp

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

func TestPickBestFloatBoundedMatchesExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	type dataset struct {
		name   string
		values []float32
	}
	datasets := []dataset{
		{name: "tie-constant", values: make([]float32, 257)},
		{name: "all-exception", values: specialExceptionFloats(257)},
		{name: "special", values: []float32{
			0, math.Float32frombits(floatNegativeZeroBits),
			math.Float32frombits(1), math.Float32frombits(0x7f7fffff),
			math.Float32frombits(0xff7fffff), math.Float32frombits(0x7fc00001),
			float32(math.Inf(1)), float32(math.Inf(-1)), 0.1, -0.1,
		}},
	}
	for seed := range 24 {
		values := make([]float32, 1+rng.Intn(1024))
		for i := range values {
			switch rng.Intn(5) {
			case 0:
				values[i] = float32(rng.Intn(200000)-100000) * 0.001
			case 1:
				values[i] = float32(rng.Intn(20000)-10000) * 0.1
			case 2:
				values[i] = math.Float32frombits(rng.Uint32())
			case 3:
				values[i] = float32(rng.Intn(1000))
			default:
				values[i] = float32(math.Pi) * float32(rng.Intn(31)-15)
			}
		}
		datasets = append(datasets, dataset{name: fmt.Sprintf("random-%d", seed), values: values})
	}

	for _, dataset := range datasets {
		t.Run(dataset.name, func(t *testing.T) {
			assertFloatPickEqual(t, dataset.values, allValidFloatPairs)
		})
	}
}

func TestPickBestFloatBoundedCandidateOrder(t *testing.T) {
	values := makeFloatTestValues(511)
	pairs := append([][2]int(nil), allValidFloatPairs...)
	rng := rand.New(rand.NewSource(2))
	rng.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	pairs = append(pairs, pairs[3], pairs[0], pairs[3], pairs[len(pairs)-1])
	assertFloatPickEqual(t, values, pairs)

	constant := make([]float32, 32)
	for i := range constant {
		constant[i] = 1.25
	}
	assertFloatPickEqual(t, constant, pairs)
}

func TestFloatPresetSearchRemainsUnbounded(t *testing.T) {
	values := makeFloatTestValues(1024)
	for n := 1; n <= 5; n++ {
		presets := allValidFloatPairs[:n]
		got := floatPickResultOf(findBestFloatParamsWithPresets(values, 0, len(values), presets))
		want := floatPickResultOf(pickBestFloat(values, 0, len(values), presets))
		if got != want {
			t.Fatalf("%d presets: got %+v, want %+v", n, got, want)
		}
	}
}

func TestSampleVectorIndexesSpanPage(t *testing.T) {
	want := []int{0, 1, 2, 3, 5, 6, 7, 9}
	for i, expected := range want {
		if got := sampleVectorIndex(i, len(want), 10); got != expected {
			t.Fatalf("sample %d: got vector %d, want %d", i, got, expected)
		}
	}
}

func TestBoundedFloatEncodingMatchesExhaustiveBytes(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	randomValues := make([]float32, 3*defaultVectorSize+17)
	for i := range randomValues {
		if i%11 == 0 {
			randomValues[i] = math.Float32frombits(rng.Uint32())
		} else {
			randomValues[i] = float32(rng.Intn(200000)-100000) * 0.001
		}
	}
	tests := []struct {
		name   string
		values []float32
	}{
		{"random-full-search", randomValues},
		{"tie", make([]float32, 2*defaultVectorSize)},
		{"all-exception", specialExceptionFloats(defaultVectorSize + 9)},
		{"special", append(specialExceptionFloats(200), randomValues[:900]...)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var encoding Encoding
			got, err := encodeFloatPage(&encoding, nil, test.values)
			if err != nil {
				t.Fatal(err)
			}
			want, err := encodeFloatPageWithPresets(&encoding, nil, test.values, allValidFloatPairs)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("encoded bytes differ: got %d bytes, want %d", len(got), len(want))
			}
		})
	}
}

func TestBoundedFloatPageSamplerMatchesExhaustive(t *testing.T) {
	first := makeSamplerFloatTestValues()
	presets := buildFloatPresetsExhaustive(first, defaultVectorSize)
	wantPresets := [][2]int{{6, 4}, {7, 7}, {8, 5}, {8, 7}}
	if !slices.Equal(presets, wantPresets) {
		t.Fatalf("sampled presets: got %v, want %v", presets, wantPresets)
	}

	var encoding Encoding
	gotPage, err := encodeFloatPage(&encoding, nil, first)
	if err != nil {
		t.Fatal(err)
	}
	wantPage, err := encodeFloatPageWithPresets(&encoding, nil, first, presets)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotPage, wantPage) {
		t.Fatal("sampled page differs from exhaustive-oracle presets")
	}

	gotStateful, err := encoding.encodeFloat(nil, first, new(encodeState))
	if err != nil {
		t.Fatal(err)
	}
	oracleState := new(encodeState)
	oracleState.floatEnc.presets = presets
	wantStateful, err := encodeFloatPageWith(&encoding, &oracleState.floatEnc, nil, first)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotStateful, wantStateful) {
		t.Fatal("stateful page encoding differs from exhaustive-oracle presets")
	}
}

type floatPickResult struct {
	e, f, exceptions int
	minEnc, maxEnc   int32
}

func floatPickResultOf(e, f, exceptions int, minEnc, maxEnc int32) floatPickResult {
	return floatPickResult{e, f, exceptions, minEnc, maxEnc}
}

func assertFloatPickEqual(t *testing.T, values []float32, pairs [][2]int) {
	t.Helper()
	got := floatPickResultOf(pickBestFloatBounded(values, 0, len(values), pairs))
	want := floatPickResultOf(pickBestFloat(values, 0, len(values), pairs))
	if got != want {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func specialExceptionFloats(n int) []float32 {
	special := [...]float32{
		math.Float32frombits(0x7fc00001),
		math.Float32frombits(0xffc12345),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		math.Float32frombits(floatNegativeZeroBits),
	}
	values := make([]float32, n)
	for i := range values {
		values[i] = special[i%len(special)]
	}
	return values
}

func buildFloatPresetsExhaustive(src []float32, vectorSize int) [][2]int {
	numVectors := (len(src) + vectorSize - 1) / vectorSize
	if numVectors <= samplerSampleVectorsPerPage {
		return nil
	}
	n := samplerSampleVectorsPerPage
	counts := make(map[[2]int]int, n)
	var sampleStorage [samplerValuesPerVector]float32
	for i := range n {
		start := sampleVectorIndex(i, n, numVectors) * vectorSize
		end := min(start+vectorSize, len(src))
		sample := subsampleFloat(sampleStorage[:], src[start:end])
		e, f, _, _, _ := pickBestFloat(sample, 0, len(sample), allValidFloatPairs)
		counts[[2]int{e, f}]++
	}
	return topPresets(counts)
}

func makeSamplerFloatTestValues() []float32 {
	values := make([]float32, 10*defaultVectorSize-37)
	categories := [...]int{0, 1, 2, 3, 4, 0, 1, 2, 4, 3}
	divisors := [...]float32{1, 10, 100, 1000, 10000}
	for i := range values {
		vector := i / defaultVectorSize
		local := i % defaultVectorSize
		values[i] = float32(local%201-100) / divisors[categories[vector]]
	}
	return values
}
