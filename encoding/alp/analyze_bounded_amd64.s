//go:build !purego

#include "textflag.h"

GLOBL boundSignBits<>(SB), RODATA|NOPTR, $32
DATA boundSignBits<>+0(SB)/8, $0x8000000080000000
DATA boundSignBits<>+8(SB)/8, $0x8000000080000000
DATA boundSignBits<>+16(SB)/8, $0x8000000080000000
DATA boundSignBits<>+24(SB)/8, $0x8000000080000000

GLOBL boundMaxInt<>(SB), RODATA|NOPTR, $32
DATA boundMaxInt<>+0(SB)/8, $0x7fffffff7fffffff
DATA boundMaxInt<>+8(SB)/8, $0x7fffffff7fffffff
DATA boundMaxInt<>+16(SB)/8, $0x7fffffff7fffffff
DATA boundMaxInt<>+24(SB)/8, $0x7fffffff7fffffff

GLOBL boundMinIntFloat<>(SB), RODATA|NOPTR, $32
DATA boundMinIntFloat<>+0(SB)/8, $0xCF000000CF000000
DATA boundMinIntFloat<>+8(SB)/8, $0xCF000000CF000000
DATA boundMinIntFloat<>+16(SB)/8, $0xCF000000CF000000
DATA boundMinIntFloat<>+24(SB)/8, $0xCF000000CF000000

// func analyzeFloatBoundedAVX2(values []float32, pe, nf, pf, ne, magic, upper float32, maxLowerBound int64, res *analyzeBoundResult)
//
// Persistent Y0/Y1 min/max accumulators are reduced only through copies. The
// lower bound is checked every 128 lanes and after the final vector.
TEXT ·analyzeFloatBoundedAVX2(SB), NOSPLIT, $0-64
	MOVQ values_base+0(FP), SI
	MOVQ values_len+8(FP), CX
	MOVQ maxLowerBound+48(FP), R12
	MOVQ res+56(FP), DI

	VBROADCASTSS pe+24(FP), Y8
	VBROADCASTSS nf+28(FP), Y9
	VBROADCASTSS pf+32(FP), Y10
	VBROADCASTSS ne+36(FP), Y11
	VBROADCASTSS magic+40(FP), Y12
	VBROADCASTSS upper+44(FP), Y13
	VMOVDQU boundSignBits<>(SB), Y15

	VMOVDQU boundMaxInt<>(SB), Y0
	VMOVDQU boundSignBits<>(SB), Y1
	XORQ R8, R8                  // valid lanes
	XORQ R14, R14                // processed lanes
	MOVQ $128, R15               // lanes until checkpoint

boundloop:
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
	VANDPS boundMaxInt<>(SB), Y3, Y4
	VCMPPS $2, Y13, Y4, Y4
	VPCMPEQD boundMinIntFloat<>(SB), Y3, Y14
	VORPS Y14, Y4, Y4
	VANDPS Y4, Y7, Y7

	VPAND Y7, Y5, Y6
	VPANDN boundMaxInt<>(SB), Y7, Y4
	VPOR Y4, Y6, Y6
	VPMINSD Y6, Y0, Y0
	VPAND Y7, Y5, Y6
	VPANDN Y15, Y7, Y4
	VPOR Y4, Y6, Y6
	VPMAXSD Y6, Y1, Y1

	VMOVMSKPS Y7, DX
	POPCNTL DX, DX
	ADDQ DX, R8
	ADDQ $32, SI
	ADDQ $8, R14
	SUBQ $8, CX
	SUBQ $8, R15
	JZ boundcheck
	TESTQ CX, CX
	JNZ boundloop

boundcheck:
	// Reduce copies so future vectors continue to update the persistent state.
	VMOVDQA Y0, Y2
	VEXTRACTI128 $1, Y2, X3
	VPMINSD X3, X2, X2
	VPSHUFD $0x4E, X2, X3
	VPMINSD X3, X2, X2
	VPSHUFD $0xB1, X2, X3
	VPMINSD X3, X2, X2
	VMOVD X2, R9

	VMOVDQA Y1, Y3
	VEXTRACTI128 $1, Y3, X4
	VPMAXSD X4, X3, X3
	VPSHUFD $0x4E, X3, X4
	VPMAXSD X4, X3, X3
	VPSHUFD $0xB1, X3, X4
	VPMAXSD X4, X3, X3
	VMOVD X3, R10

	MOVQ R14, R11
	SUBQ R8, R11                // exceptions = processed - valid
	MOVQ R11, R13
	IMULQ $48, R13              // exception contribution
	CMPQ R14, R11
	JE boundcompare              // no valid lanes: range contribution is zero
	MOVL R10, BX
	SUBL R9, BX                 // uint32(max-min), valid range is at most 2^32-1
	JZ boundcompare
	BSRQ BX, AX
	INCQ AX
	IMULQ values_len+8(FP), AX
	ADDQ AX, R13

boundcompare:
	CMPQ R13, R12
	JG boundpruned
	TESTQ CX, CX
	JZ boundcomplete
	MOVQ $128, R15
	JMP boundloop

boundpruned:
	MOVQ $1, AX
	JMP boundstore

boundcomplete:
	XORQ AX, AX

boundstore:
	MOVQ R11, 0(DI)
	MOVL R9, 8(DI)
	MOVL R10, 12(DI)
	MOVQ R14, 16(DI)
	MOVQ AX, 24(DI)
	VZEROUPPER
	RET
