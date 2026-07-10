package parquet

import (
	"bytes"
	"fmt"
	"testing"

	parquetencoding "github.com/parquet-go/parquet-go/encoding"
)

type testStatefulEncodingFactory struct {
	delegatedEncoding
	result    parquetencoding.ResettableEncoding
	hasResult bool
	instances []*testStatefulEncoding
	encodes   int
}

func (f *testStatefulEncodingFactory) NewStatefulEncoder() parquetencoding.ResettableEncoding {
	if f.hasResult {
		return f.result
	}
	encoding := &testStatefulEncoding{delegatedEncoding: f.delegatedEncoding}
	f.instances = append(f.instances, encoding)
	return encoding
}

func (f *testStatefulEncodingFactory) EncodeDouble(dst []byte, src []float64) ([]byte, error) {
	f.encodes++
	return f.delegatedEncoding.EncodeDouble(dst, src)
}

type delegatedEncoding interface{ parquetencoding.Encoding }

type testStatefulEncoding struct {
	delegatedEncoding
	encodes int
	resets  int
	onReset func()
}

func (e *testStatefulEncoding) EncodeDouble(dst []byte, src []float64) ([]byte, error) {
	e.encodes++
	return e.delegatedEncoding.EncodeDouble(dst, src)
}

func (e *testStatefulEncoding) Reset() {
	e.resets++
	if e.onReset != nil {
		e.onReset()
	}
}

func TestNewColumnEncodingRejectsInvalidFactoryResult(t *testing.T) {
	tests := []struct {
		name    string
		result  parquetencoding.ResettableEncoding
		wantErr string
	}{
		{
			name:    "nil",
			wantErr: "parquet: stateful encoding factory for PLAIN returned nil",
		},
		{
			name:    "different encoding",
			result:  &testStatefulEncoding{delegatedEncoding: &ALP},
			wantErr: "parquet: stateful encoding factory for PLAIN returned encoding ALP",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			factory := &testStatefulEncodingFactory{delegatedEncoding: &Plain, result: test.result, hasResult: true}
			defer func() {
				if got := fmt.Sprint(recover()); got != test.wantErr {
					t.Fatalf("panic = %q, want %q", got, test.wantErr)
				}
			}()
			newColumnEncoding(factory)
		})
	}
}

func TestColumnWriterDoesNotResetConfiguredResettableEncoding(t *testing.T) {
	type row struct{ Value float64 }

	shared := &testStatefulEncoding{delegatedEncoding: &Plain}
	var output bytes.Buffer
	w := NewGenericWriter[row](&output, DefaultEncodingFor(Double, shared))
	column := w.base.writer.currentRowGroup.columns[0]
	if column.encoding != shared || column.originalEncoding != shared {
		t.Fatal("column does not retain the configured encoding")
	}
	if column.originalEncodingOwned {
		t.Fatal("configured resettable encoding is marked as writer-owned")
	}

	if _, err := w.Write([]row{{Value: 1.25}}); err != nil {
		t.Fatal(err)
	}
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}
	if shared.resets != 0 {
		t.Fatalf("configured resettable encoding reset %d times, want 0", shared.resets)
	}
}

func TestColumnWriterUsesAndResetsEffectiveEncoding(t *testing.T) {
	type row struct {
		A float64
		B float64
	}

	factory := &testStatefulEncodingFactory{delegatedEncoding: &Plain}
	var output bytes.Buffer
	w := NewGenericWriter[row](&output, DefaultEncodingFor(Double, factory))
	columns := w.base.writer.currentRowGroup.columns
	if len(factory.instances) != len(columns) {
		t.Fatalf("created %d encoding instances for %d columns", len(factory.instances), len(columns))
	}
	for i, column := range columns {
		instance := factory.instances[i]
		if column.encoding != instance || column.originalEncoding != instance {
			t.Fatalf("column %d does not consistently store its effective encoding", i)
		}
		if !column.originalEncodingOwned {
			t.Fatalf("column %d effective encoding is not marked as writer-owned", i)
		}
		instance.onReset = func() {
			if column.encoding != column.originalEncoding {
				t.Errorf("column %d reset before restoring its original encoding", i)
			}
		}
	}
	// Configuration probes compatibility on the descriptor before leaf instances
	// are created; only payload encoding below is relevant to this assertion.
	factory.encodes = 0

	if _, err := w.Write([]row{{A: 1.25, B: 2.5}, {A: 3.75, B: 5}}); err != nil {
		t.Fatal(err)
	}
	for _, column := range columns {
		if err := column.Flush(); err != nil {
			t.Fatal(err)
		}
	}
	columns[0].encoding = &Plain
	columns[0].hasSwitchedToPlain = true
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}
	for i, column := range columns {
		if column.encoding != factory.instances[i] || column.hasSwitchedToPlain {
			t.Fatalf("column %d did not restore its effective encoding", i)
		}
	}
	firstRowGroupEncodes := factory.instances[0].encodes
	if _, err := w.Write([]row{{A: 6.25, B: 7.5}}); err != nil {
		t.Fatal(err)
	}
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}

	if factory.encodes != 0 {
		t.Fatalf("configured descriptor encoded %d pages", factory.encodes)
	}
	for i, instance := range factory.instances {
		if instance.encodes == 0 {
			t.Errorf("effective encoding for column %d was not used", i)
		}
		if instance.resets != 2 {
			t.Errorf("effective encoding for column %d reset %d times, want 2", i, instance.resets)
		}
	}
	if factory.instances[0].encodes <= firstRowGroupEncodes {
		t.Error("effective encoding was not reused for the next row group")
	}
}
