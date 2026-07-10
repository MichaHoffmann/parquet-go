package alp

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestPickBestDoubleBoundedMatchesExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	type dataset struct {
		name   string
		values []float64
	}
	datasets := []dataset{
		{name: "tie-constant", values: make([]float64, 257)},
		{name: "all-exception", values: specialExceptionDoubles(257)},
		{name: "special", values: []float64{
			0, math.Float64frombits(doubleNegativeZeroBits),
			math.Float64frombits(1), math.MaxFloat64, -math.MaxFloat64,
			math.Float64frombits(0x7ff8000000000001),
			math.Inf(1), math.Inf(-1), 0.1, -0.1,
		}},
	}
	for seed := range 24 {
		values := make([]float64, 1+rng.Intn(defaultVectorSize))
		for i := range values {
			switch rng.Intn(5) {
			case 0:
				values[i] = float64(rng.Intn(200000)-100000) * 0.001
			case 1:
				values[i] = float64(rng.Intn(20000)-10000) * 0.1
			case 2:
				values[i] = math.Float64frombits(rng.Uint64())
			case 3:
				values[i] = float64(rng.Intn(1000))
			default:
				values[i] = math.Pi * float64(rng.Intn(31)-15)
			}
		}
		datasets = append(datasets, dataset{name: fmt.Sprintf("random-%d", seed), values: values})
	}

	for _, dataset := range datasets {
		t.Run(dataset.name, func(t *testing.T) {
			assertDoublePickEqual(t, dataset.values, allValidDoublePairs)
		})
	}
}

func TestPickBestDoubleBoundedCandidateOrder(t *testing.T) {
	values := makeDoubleTestValues(511)
	pairs := append([][2]int(nil), allValidDoublePairs...)
	rng := rand.New(rand.NewSource(5))
	rng.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	pairs = append(pairs, pairs[3], pairs[0], pairs[3], pairs[len(pairs)-1])
	assertDoublePickEqual(t, values, pairs)

	constant := make([]float64, 32)
	for i := range constant {
		constant[i] = 1.25
	}
	assertDoublePickEqual(t, constant, pairs)
}

func TestDoublePresetSearchRemainsUnbounded(t *testing.T) {
	values := makeDoubleTestValues(defaultVectorSize)
	for n := 1; n <= maxPresetCombinations; n++ {
		presets := allValidDoublePairs[:n]
		got := doublePickResultOf(findBestDoubleParamsWithPresets(values, 0, len(values), presets))
		want := doublePickResultOf(pickBestDouble(values, 0, len(values), presets))
		if got != want {
			t.Fatalf("%d presets: got %+v, want %+v", n, got, want)
		}
	}
}

type doublePickResult struct {
	e, f, exceptions int
	minEnc, maxEnc   int64
}

func doublePickResultOf(e, f, exceptions int, minEnc, maxEnc int64) doublePickResult {
	return doublePickResult{e, f, exceptions, minEnc, maxEnc}
}

func assertDoublePickEqual(t *testing.T, values []float64, pairs [][2]int) {
	t.Helper()
	got := doublePickResultOf(pickBestDoubleBounded(values, 0, len(values), pairs))
	want := doublePickResultOf(pickBestDouble(values, 0, len(values), pairs))
	if got != want {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func specialExceptionDoubles(n int) []float64 {
	special := [...]float64{
		math.Float64frombits(0x7ff8000000000001),
		math.Float64frombits(0xfff8123456789abc),
		math.Inf(1),
		math.Inf(-1),
		math.Float64frombits(doubleNegativeZeroBits),
	}
	values := make([]float64, n)
	for i := range values {
		values[i] = special[i%len(special)]
	}
	return values
}

func makeDoubleTestValues(n int) []float64 {
	values := make([]float64, n)
	for i := range values {
		switch i % 5 {
		case 0:
			values[i] = float64(i%1000) * 0.01
		case 1:
			values[i] = -float64(i%700) * 0.1
		case 2:
			values[i] = math.Pi * float64(i%17)
		case 3:
			values[i] = float64(i % 100)
		default:
			values[i] = float64(i%300) * 0.001
		}
	}
	return values
}
