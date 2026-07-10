package alp

// Wire constants and numeric limits follow the parquet-java reference
// implementation.

const (
	// Page header field values.
	alpCompressionMode     = 0
	alpIntegerEncodingFOR  = 0
	alpHeaderSize          = 7 // compression_mode(1)+integer_encoding(1)+log_vector_size(1)+num_elements(4)
	defaultVectorSize      = 1024
	defaultVectorSizeLog   = 10
	minLogVectorSize       = 3
	maxLogVectorSize       = 15
	floatMaxExponent       = 10
	doubleMaxExponent      = 18
	alpInfoSize            = 4 // exponent(1)+factor(1)+num_exceptions(2)
	floatForInfoSize       = 5 // frame_of_reference(4)+bit_width(1)
	doubleForInfoSize      = 9 // frame_of_reference(8)+bit_width(1)
	floatNegativeZeroBits  = 0x80000000
	doubleNegativeZeroBits = 0x8000000000000000
)

// Preset-search limits matching the current draft implementations.
const (
	samplerValuesPerVector      = 256 // number of values collected from each sample vector
	samplerSampleVectorsPerPage = 8   // number of sample vectors collected
	maxPresetCombinations       = 5   // number of preset (exponent, factor) pairs kept
)

// Magic numbers for the fast-rounding trick (see ALP paper, Section 3.2).
const (
	magicFloat  = float32(12_582_912.0)            // 2^22 + 2^23
	magicDouble = float64(6_755_399_441_055_744.0) // 2^51 + 2^52
)

// Encoding limits: values outside this range after scaling become exceptions.
const (
	doubleEncodingUpperLimit = 9223372036854774784.0
	doubleEncodingLowerLimit = -9223372036854775808.0
	floatEncodingUpperLimit  = float32(2147483520.0)
	floatEncodingLowerLimit  = float32(-2147483648.0)
)

// POW10: positive powers used for scaling up during encode/decode.
//
//	Encode: fastRound(value * POW10[e] * POW10_NEGATIVE[f])
//	Decode: encoded * POW10[f] * POW10_NEGATIVE[e]
var floatPow10 = [...]float32{1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10}

var doublePow10 = [...]float64{
	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18,
}

// POW10_NEGATIVE: reciprocals used for scaling down (multiply-by-reciprocal).
// Using separate negative-power arrays instead of division ensures C++ wire
// compatibility.
var floatPow10Negative = [...]float32{
	1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10,
}

var doublePow10Negative = [...]float64{
	1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
	1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18,
}
