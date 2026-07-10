package alp

import (
	"bytes"
	"math"
	"testing"
)

func TestEncodeStateClearsPresetsForSmallPageAndRetainsScratch(t *testing.T) {
	var e Encoding
	var state encodeState
	largeFloats := makeSamplerFloatTestValues()
	largeDoubles := make([]float64, len(largeFloats))
	for i := range largeFloats {
		largeDoubles[i] = float64(largeFloats[i])
	}

	if _, err := e.encodeFloat(nil, largeFloats, &state); err != nil {
		t.Fatal(err)
	}
	if state.floatEnc.presets == nil {
		t.Fatal("large float page did not select presets")
	}
	smallFloats := make([]float32, samplerSampleVectorsPerPage*defaultVectorSize)
	smallFloats[len(smallFloats)-1] = float32(math.NaN())
	gotFloat, err := e.encodeFloat(nil, smallFloats, &state)
	if err != nil {
		t.Fatal(err)
	}
	wantFloat, err := e.EncodeFloat(nil, smallFloats)
	if err != nil {
		t.Fatal(err)
	}
	if state.floatEnc.presets != nil {
		t.Fatal("small float page retained presets")
	}
	if !bytes.Equal(gotFloat, wantFloat) {
		t.Fatal("small float page did not use full search")
	}

	if _, err := e.encodeDouble(nil, largeDoubles, &state); err != nil {
		t.Fatal(err)
	}
	if state.doubleEnc.presets == nil {
		t.Fatal("large double page did not select presets")
	}
	smallDoubles := make([]float64, samplerSampleVectorsPerPage*defaultVectorSize)
	smallDoubles[len(smallDoubles)-1] = math.NaN()
	gotDouble, err := e.encodeDouble(nil, smallDoubles, &state)
	if err != nil {
		t.Fatal(err)
	}
	wantDouble, err := e.EncodeDouble(nil, smallDoubles)
	if err != nil {
		t.Fatal(err)
	}
	if state.doubleEnc.presets != nil {
		t.Fatal("small double page retained presets")
	}
	if !bytes.Equal(gotDouble, wantDouble) {
		t.Fatal("small double page did not use full search")
	}

	floatEncoded := &state.floatEnc.encoded[0]
	floatExceptions := &state.floatEnc.excPos[0]
	doubleEncoded := &state.doubleEnc.encoded[0]
	doubleExceptions := &state.doubleEnc.excPos[0]
	state.Reset()
	if state.floatEnc.presets != nil || state.doubleEnc.presets != nil {
		t.Fatal("reset retained page presets")
	}
	if floatEncoded != &state.floatEnc.encoded[0] || floatExceptions != &state.floatEnc.excPos[0] ||
		doubleEncoded != &state.doubleEnc.encoded[0] || doubleExceptions != &state.doubleEnc.excPos[0] {
		t.Fatal("reset discarded reusable scratch buffers")
	}
}
