//go:build !purego

#include "textflag.h"

// Integer mask constants, 8 lanes each (256-bit).
GLOBL signBits<>(SB), RODATA|NOPTR, $32
DATA signBits<>+0(SB)/4, $0x80000000
DATA signBits<>+4(SB)/4, $0x80000000
DATA signBits<>+8(SB)/4, $0x80000000
DATA signBits<>+12(SB)/4, $0x80000000
DATA signBits<>+16(SB)/4, $0x80000000
DATA signBits<>+20(SB)/4, $0x80000000
DATA signBits<>+24(SB)/4, $0x80000000
DATA signBits<>+28(SB)/4, $0x80000000

GLOBL nanExp<>(SB), RODATA|NOPTR, $32
DATA nanExp<>+0(SB)/4, $0x7F800000
DATA nanExp<>+4(SB)/4, $0x7F800000
DATA nanExp<>+8(SB)/4, $0x7F800000
DATA nanExp<>+12(SB)/4, $0x7F800000
DATA nanExp<>+16(SB)/4, $0x7F800000
DATA nanExp<>+20(SB)/4, $0x7F800000
DATA nanExp<>+24(SB)/4, $0x7F800000
DATA nanExp<>+28(SB)/4, $0x7F800000

GLOBL maxInt<>(SB), RODATA|NOPTR, $32
DATA maxInt<>+0(SB)/4, $0x7FFFFFFF
DATA maxInt<>+4(SB)/4, $0x7FFFFFFF
DATA maxInt<>+8(SB)/4, $0x7FFFFFFF
DATA maxInt<>+12(SB)/4, $0x7FFFFFFF
DATA maxInt<>+16(SB)/4, $0x7FFFFFFF
DATA maxInt<>+20(SB)/4, $0x7FFFFFFF
DATA maxInt<>+24(SB)/4, $0x7FFFFFFF
DATA maxInt<>+28(SB)/4, $0x7FFFFFFF

GLOBL minIntFloat<>(SB), RODATA|NOPTR, $32
DATA minIntFloat<>+0(SB)/4, $0xCF000000
DATA minIntFloat<>+4(SB)/4, $0xCF000000
DATA minIntFloat<>+8(SB)/4, $0xCF000000
DATA minIntFloat<>+12(SB)/4, $0xCF000000
DATA minIntFloat<>+16(SB)/4, $0xCF000000
DATA minIntFloat<>+20(SB)/4, $0xCF000000
DATA minIntFloat<>+24(SB)/4, $0xCF000000
DATA minIntFloat<>+28(SB)/4, $0xCF000000

// func analyzeFloatAVX2(values []float32, pe, nf, pf, ne, magic, upper float32, res *analyzeResult)
//
// Processes len(values) float32 lanes (a non-zero multiple of 8), bit-exactly
// equivalent to analyzeFloatScalar for the same (exponent, factor). Writes
// exceptions/minEnc/maxEnc to *res.
TEXT ·analyzeFloatAVX2(SB), NOSPLIT, $0-56
	MOVQ values_base+0(FP), SI
	MOVQ values_len+8(FP), DI  // DI = total lane count (multiple of 8)
	MOVQ res+48(FP), AX

	// Broadcast the scalar factors into the persistent constant registers.
	VBROADCASTSS pe+24(FP), Y8
	VBROADCASTSS nf+28(FP), Y9
	VBROADCASTSS pf+32(FP), Y10
	VBROADCASTSS ne+36(FP), Y11
	VBROADCASTSS magic+40(FP), Y12
	VBROADCASTSS upper+44(FP), Y13
	VMOVDQU signBits<>(SB), Y15

	// Accumulators: Y0 = running min (init MaxInt32), Y1 = running max (init
	// MinInt32), R8 = count of valid (non-exception) lanes.
	VMOVDQU maxInt<>(SB), Y0
	VMOVDQU signBits<>(SB), Y1  // MinInt32 == 0x80000000
	XORQ R8, R8
	MOVQ DI, CX  // CX = remaining lanes

loop:
	VMOVUPS (SI), Y2            // Y2 = value
	VMULPS  Y8, Y2, Y3         // Y3 = value * pe
	VMULPS  Y9, Y3, Y3         // Y3 = scaled = value * pe * nf

	// Fast-round: m = (scaled & signbit) | magic ; r = (scaled + m) - m
	VANDPS  Y15, Y3, Y4        // Y4 = sign bit of scaled
	VORPS   Y12, Y4, Y4        // Y4 = magic carrying scaled's sign
	VADDPS  Y4, Y3, Y5         // Y5 = scaled + m
	VSUBPS  Y4, Y5, Y5         // Y5 = rounded float
	VCVTTPS2DQ Y5, Y5          // Y5 = enc (int32, truncation of integral value)

	// Decode: decoded = float(enc) * pf * ne
	VCVTDQ2PS Y5, Y6           // Y6 = float(enc)
	VMULPS  Y10, Y6, Y6        // Y6 *= pf
	VMULPS  Y11, Y6, Y6        // Y6 = decoded

	// eqMask (Y7): bit-exact round-trip match value == decoded
	VPCMPEQD Y2, Y6, Y7

	// rangeValid: abs(scaled) <= upper, plus the exact signed int32 minimum.
	VANDPS  maxInt<>(SB), Y3, Y4
	VCMPPS  $2, Y13, Y4, Y4
	VPCMPEQD minIntFloat<>(SB), Y3, Y14
	VORPS   Y14, Y4, Y4
	VANDPS  Y4, Y7, Y7         // Y7 = eq AND inRange

	// Y7 is now the complete valid mask. NaN/Inf fail the ordered range
	// comparisons, while -0.0 fails the bit-exact decoded-value comparison.

	// min over valid lanes: encMin = (enc & valid) | (MaxInt32 & ~valid)
	VPAND   Y7, Y5, Y6
	VPANDN  maxInt<>(SB), Y7, Y4   // Y4 = ~valid & MaxInt32
	VPOR    Y4, Y6, Y6
	VPMINSD Y6, Y0, Y0

	// max over valid lanes: encMax = (enc & valid) | (MinInt32 & ~valid)
	VPAND   Y7, Y5, Y6
	VPANDN  Y15, Y7, Y4           // Y4 = ~valid & MinInt32
	VPOR    Y4, Y6, Y6
	VPMAXSD Y6, Y1, Y1

	// Accumulate valid-lane count.
	VMOVMSKPS Y7, DX
	POPCNTL DX, DX
	ADDQ    DX, R8

	ADDQ $32, SI
	SUBQ $8, CX
	JNZ  loop

	// Horizontal min reduction of Y0 -> scalar in DX.
	VEXTRACTI128 $1, Y0, X2
	VPMINSD X2, X0, X0
	VPSHUFD $0x4E, X0, X2
	VPMINSD X2, X0, X0
	VPSHUFD $0xB1, X0, X2
	VPMINSD X2, X0, X0
	VMOVD   X0, DX

	// Horizontal max reduction of Y1 -> scalar in BX.
	VEXTRACTI128 $1, Y1, X2
	VPMAXSD X2, X1, X1
	VPSHUFD $0x4E, X1, X2
	VPMAXSD X2, X1, X1
	VPSHUFD $0xB1, X1, X2
	VPMAXSD X2, X1, X1
	VMOVD   X1, BX

	// exceptions = total - valid
	SUBQ R8, DI

	MOVQ DI, (AX)      // res.exceptions (int64)
	MOVL DX, 8(AX)     // res.minEnc (int32)
	MOVL BX, 12(AX)    // res.maxEnc (int32)
	VZEROUPPER
	RET

// func analyzeFloatEncodedAVX2(values []float32, pe, nf, pf, ne, magic, upper float32, encoded []uint32, excPos []uint16, res *analyzeResult)
//
// analyzeFloatAVX2 plus materialization of each raw encoded lane and compaction
// of exception positions. Used when there is exactly one preset, so packing can
// reuse this work instead of repeating the floating-point operations.
TEXT ·analyzeFloatEncodedAVX2(SB), NOSPLIT, $0-104
	MOVQ values_base+0(FP), SI
	MOVQ values_len+8(FP), DI
	MOVQ encoded_base+48(FP), R9
	MOVQ excPos_base+72(FP), R12

	VBROADCASTSS pe+24(FP), Y8
	VBROADCASTSS nf+28(FP), Y9
	VBROADCASTSS pf+32(FP), Y10
	VBROADCASTSS ne+36(FP), Y11
	VBROADCASTSS magic+40(FP), Y12
	VBROADCASTSS upper+44(FP), Y13
	VMOVDQU signBits<>(SB), Y15

	VMOVDQU maxInt<>(SB), Y0
	VMOVDQU signBits<>(SB), Y1
	XORQ R11, R11
	XORQ R13, R13
	MOVQ DI, CX

analyzeencodedloop:
	VMOVUPS (SI), Y2
	VMULPS Y8, Y2, Y3
	VMULPS Y9, Y3, Y3
	VANDPS Y15, Y3, Y4
	VORPS Y12, Y4, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y4, Y5, Y5
	VCVTTPS2DQ Y5, Y5

	VCVTDQ2PS Y5, Y6
	VMULPS Y10, Y6, Y6
	VMULPS Y11, Y6, Y6
	VPCMPEQD Y2, Y6, Y7
	VANDPS maxInt<>(SB), Y3, Y4
	VCMPPS $2, Y13, Y4, Y4
	VPCMPEQD minIntFloat<>(SB), Y3, Y14
	VORPS Y14, Y4, Y4
	VANDPS Y4, Y7, Y7

	VMOVDQU Y5, (R9)
	ADDQ $32, R9
	VMOVMSKPS Y7, DX
	MOVL DX, BX
	POPCNTL BX, BX
	ADDQ BX, R11
	NOTL DX
	ANDL $0xFF, DX

analyzeencodedcompact:
	TESTL DX, DX
	JZ analyzeencodedcompactdone
	BSFL DX, BX
	LEAQ (R13)(BX*1), AX
	MOVW AX, (R12)
	ADDQ $2, R12
	LEAL -1(DX), AX
	ANDL AX, DX
	JMP analyzeencodedcompact

analyzeencodedcompactdone:
	ADDQ $8, R13

	VPAND Y7, Y5, Y6
	VPANDN maxInt<>(SB), Y7, Y4
	VPOR Y4, Y6, Y6
	VPMINSD Y6, Y0, Y0
	VPAND Y7, Y5, Y6
	VPANDN Y15, Y7, Y4
	VPOR Y4, Y6, Y6
	VPMAXSD Y6, Y1, Y1

	ADDQ $32, SI
	SUBQ $8, CX
	JNZ analyzeencodedloop

	VEXTRACTI128 $1, Y0, X2
	VPMINSD X2, X0, X0
	VPSHUFD $0x4E, X0, X2
	VPMINSD X2, X0, X0
	VPSHUFD $0xB1, X0, X2
	VPMINSD X2, X0, X0
	VMOVD X0, DX

	VEXTRACTI128 $1, Y1, X2
	VPMAXSD X2, X1, X1
	VPSHUFD $0x4E, X1, X2
	VPMAXSD X2, X1, X1
	VPSHUFD $0xB1, X1, X2
	VPMAXSD X2, X1, X1
	VMOVD X1, BX

	SUBQ R11, DI
	MOVQ res+96(FP), AX
	MOVQ DI, (AX)
	MOVL DX, 8(AX)
	MOVL BX, 12(AX)
	VZEROUPPER
	RET
// func encodeFloatAVX2(values []float32, pe, nf, pf, ne, magic, upper float32, minEnc int32, deltas []uint32, excPos []uint16)
//
// Processes len(values) float32 lanes (a non-zero multiple of 8). For each lane
// it writes encoded-minEnc (or zero for an exception) to deltas and compacts
// exception positions into excPos. The encoded value and exception decision are
// computed exactly as in evalFloat, so this is bit-exactly equivalent to
// encodeFloatDeltasScalar.
TEXT ·encodeFloatAVX2(SB), NOSPLIT, $0-104
	MOVQ values_base+0(FP), SI
	MOVQ values_len+8(FP), CX   // CX = remaining lanes (multiple of 8)
	MOVQ deltas_base+56(FP), R8 // R8 = delta write pointer
	MOVQ excPos_base+80(FP), R9 // R9 = exception-position write pointer

	VBROADCASTSS pe+24(FP), Y8
	VBROADCASTSS nf+28(FP), Y9
	VBROADCASTSS pf+32(FP), Y10
	VBROADCASTSS ne+36(FP), Y11
	VBROADCASTSS magic+40(FP), Y12
	VBROADCASTSS upper+44(FP), Y13
	VMOVDQU signBits<>(SB), Y15
	MOVL minEnc+48(FP), AX
	VMOVD AX, X0
	VPBROADCASTD X0, Y0
	XORQ R10, R10               // R10 = base lane index

encloop:
	VMOVUPS (SI), Y2            // Y2 = value
	VMULPS  Y8, Y2, Y3         // Y3 = value * pe
	VMULPS  Y9, Y3, Y3         // Y3 = scaled = value * pe * nf

	// Fast-round: m = (scaled & signbit) | magic ; r = (scaled + m) - m
	VANDPS  Y15, Y3, Y4
	VORPS   Y12, Y4, Y4
	VADDPS  Y4, Y3, Y5
	VSUBPS  Y4, Y5, Y5
	VCVTTPS2DQ Y5, Y5          // Y5 = enc (int32)

	// Decode: decoded = float(enc) * pf * ne
	VCVTDQ2PS Y5, Y6
	VMULPS  Y10, Y6, Y6
	VMULPS  Y11, Y6, Y6

	// eqMask (Y7): value == decoded
	VPCMPEQD Y2, Y6, Y7

	// rangeValid: abs(scaled) <= upper, plus the exact signed int32 minimum.
	VANDPS  maxInt<>(SB), Y3, Y4
	VCMPPS  $2, Y13, Y4, Y4
	VPCMPEQD minIntFloat<>(SB), Y3, Y14
	VORPS   Y14, Y4, Y4
	VANDPS  Y4, Y7, Y7

	// Y7 is now the complete valid mask; see analyzeFloatAVX2 above.

	// Store FOR deltas, zeroing exception lanes.
	VPSUBD  Y0, Y5, Y5
	VPAND   Y7, Y5, Y5
	VMOVDQU Y5, (R8)
	ADDQ    $32, R8

	// Compact exception positions from the inverse valid mask.
	VMOVMSKPS Y7, DX
	NOTL    DX
	ANDL    $0xFF, DX

compactexceptions:
	TESTL DX, DX
	JZ compactdone
	BSFL DX, BX
	LEAQ (R10)(BX*1), AX
	MOVW AX, (R9)
	ADDQ $2, R9
	LEAL -1(DX), AX
	ANDL AX, DX
	JMP compactexceptions

compactdone:
	ADDQ $8, R10

	ADDQ $32, SI
	SUBQ $8, CX
	JNZ  encloop

	VZEROUPPER
	RET
