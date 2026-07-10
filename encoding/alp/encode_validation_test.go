package alp

import (
	"math"
	"testing"
)

func TestValidateEncodeCount(t *testing.T) {
	var e Encoding
	if err := validateEncodeCount(&e, 1<<31-1); err != nil {
		t.Fatalf("maximum count rejected: %v", err)
	}
	if ^uint(0)>>32 != 0 {
		if err := validateEncodeCount(&e, int64ToInt(1<<31)); err == nil {
			t.Fatal("count exceeding int32 was accepted")
		}
	}
}

func int64ToInt(v int64) int { return int(v) }

func TestValidateVectorOutputSize(t *testing.T) {
	maxUint32 := uint64(math.MaxUint32)
	if uint64(^uint(0)>>1) < maxUint32 {
		t.Skip("int cannot represent a uint32-sized output")
	}
	var e Encoding
	if err := validateVectorOutputSize(&e, []byte{0}, 0, int(maxUint32)); err == nil {
		t.Fatal("offset overflow was accepted")
	}
}
