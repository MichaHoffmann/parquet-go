package alp_test

import (
	"testing"

	"github.com/parquet-go/parquet-go/encoding/alp"
	"github.com/parquet-go/parquet-go/encoding/fuzz"
)

// The shared fuzz harness treats NaN != NaN. Dedicated round-trip tests cover
// bit-exact NaNs, infinities, and negative zero.
func FuzzEncodeFloat(f *testing.F) {
	fuzz.EncodeFloat(f, new(alp.Encoding))
}

func FuzzEncodeDouble(f *testing.F) {
	fuzz.EncodeDouble(f, new(alp.Encoding))
}
