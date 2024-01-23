pub struct NArray<T> {
    array: Vec<T>,
    pub dim: Dimension,
}

pub fn build<T: Clone>(dim: &Dimension, init_value: T) -> NArray<T> {
    let size = match *dim {
        Dimension::D1 { n0 } => n0,
        Dimension::D2 { n0, n1 } => n0 * n1,
        Dimension::D3 { n0, n1, n2, .. } => n0 * n1 * n2,
        Dimension::D4 { n0, n1, n2, n3, .. } => n0 * n1 * n2 * n3,
        Dimension::D5 {
            n0, n1, n2, n3, n4, ..
        } => n0 * n1 * n2 * n3 * n4,
        Dimension::D6 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            ..
        } => n0 * n1 * n2 * n3 * n4 * n5,
        Dimension::D7 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            ..
        } => n0 * n1 * n2 * n3 * n4 * n5 * n6,
        Dimension::D8 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            n7,
            ..
        } => n0 * n1 * n2 * n3 * n4 * n5 * n6 * n7,
        Dimension::D9 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            n7,
            n8,
            ..
        } => n0 * n1 * n2 * n3 * n4 * n5 * n6 * n7 * n8,
        Dimension::D10 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            n7,
            n8,
            n9,
            ..
        } => n0 * n1 * n2 * n3 * n4 * n5 * n6 * n7 * n8 * n9,
        Dimension::D11 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            n7,
            n8,
            n9,
            n10,
            ..
        } => n0 * n1 * n2 * n3 * n4 * n5 * n6 * n7 * n8 * n9 * n10,
    };

    NArray {
        array: vec![init_value; size],
        dim: dim.clone(),
    }
}

// row major idx formula
// idx.0 * (N1*N2*N3*...) + idx.1 * (N2*N3*...) + ... + idx.k
#[derive(Clone)]
pub enum Index {
    I1(usize),
    I2(usize, usize),
    I3(usize, usize, usize),
    I4(usize, usize, usize, usize),
    I5(usize, usize, usize, usize, usize),
    I6(usize, usize, usize, usize, usize, usize),
    I7(usize, usize, usize, usize, usize, usize, usize),
    I8(usize, usize, usize, usize, usize, usize, usize, usize),
    I9(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ),
    I10(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ),
    I11(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ),
}

#[derive(Clone)]
pub enum Dimension {
    D1 {
        n0: usize,
    },
    D2 {
        n0: usize,
        n1: usize,
    },
    D3 {
        n0: usize,
        n1: usize,
        n2: usize,
        _n1n2: usize,
    },
    D4 {
        n0: usize,
        n1: usize,
        n2: usize,
        n3: usize,
        _n1n2n3: usize,
        _n2n3: usize,
    },
    D5 {
        n0: usize,
        n1: usize,
        n2: usize,
        n3: usize,
        n4: usize,
        _n1n2n3n4: usize,
        _n2n3n4: usize,
        _n3n4: usize,
    },
    D6 {
        n0: usize,
        n1: usize,
        n2: usize,
        n3: usize,
        n4: usize,
        n5: usize,
        _n1n2n3n4n5: usize,
        _n2n3n4n5: usize,
        _n3n4n5: usize,
        _n4n5: usize,
    },
    D7 {
        n0: usize,
        n1: usize,
        n2: usize,
        n3: usize,
        n4: usize,
        n5: usize,
        n6: usize,
        _n1n2n3n4n5n6: usize,
        _n2n3n4n5n6: usize,
        _n3n4n5n6: usize,
        _n4n5n6: usize,
        _n5n6: usize,
    },
    D8 {
        n0: usize,
        n1: usize,
        n2: usize,
        n3: usize,
        n4: usize,
        n5: usize,
        n6: usize,
        n7: usize,
        _n1n2n3n4n5n6n7: usize,
        _n2n3n4n5n6n7: usize,
        _n3n4n5n6n7: usize,
        _n4n5n6n7: usize,
        _n5n6n7: usize,
        _n6n7: usize,
    },
    D9 {
        n0: usize,
        n1: usize,
        n2: usize,
        n3: usize,
        n4: usize,
        n5: usize,
        n6: usize,
        n7: usize,
        n8: usize,
        _n1n2n3n4n5n6n7n8: usize,
        _n2n3n4n5n6n7n8: usize,
        _n3n4n5n6n7n8: usize,
        _n4n5n6n7n8: usize,
        _n5n6n7n8: usize,
        _n6n7n8: usize,
        _n7n8: usize,
    },
    D10 {
        n0: usize,
        n1: usize,
        n2: usize,
        n3: usize,
        n4: usize,
        n5: usize,
        n6: usize,
        n7: usize,
        n8: usize,
        n9: usize,
        _n1n2n3n4n5n6n7n8n9: usize,
        _n2n3n4n5n6n7n8n9: usize,
        _n3n4n5n6n7n8n9: usize,
        _n4n5n6n7n8n9: usize,
        _n5n6n7n8n9: usize,
        _n6n7n8n9: usize,
        _n7n8n9: usize,
        _n8n9: usize,
    },
    D11 {
        n0: usize,
        n1: usize,
        n2: usize,
        n3: usize,
        n4: usize,
        n5: usize,
        n6: usize,
        n7: usize,
        n8: usize,
        n9: usize,
        n10: usize,
        _n1n2n3n4n5n6n7n8n9n10: usize,
        _n2n3n4n5n6n7n8n9n10: usize,
        _n3n4n5n6n7n8n9n10: usize,
        _n4n5n6n7n8n9n10: usize,
        _n5n6n7n8n9n10: usize,
        _n6n7n8n9n10: usize,
        _n7n8n9n10: usize,
        _n8n9n10: usize,
        _n9n10: usize,
    },
}

impl Index {
    pub fn new(index: &[usize]) -> Index {
        match index {
            [i0] => Index::I1(*i0),
            [i0, i1] => Index::I2(*i0, *i1),
            [i0, i1, i2] => Index::I3(*i0, *i1, *i2),
            [i0, i1, i2, i3] => Index::I4(*i0, *i1, *i2, *i3),
            [i0, i1, i2, i3, i4] => Index::I5(*i0, *i1, *i2, *i3, *i4),
            [i0, i1, i2, i3, i4, i5] => Index::I6(*i0, *i1, *i2, *i3, *i4, *i5),
            [i0, i1, i2, i3, i4, i5, i6] => Index::I7(*i0, *i1, *i2, *i3, *i4, *i5, *i6),
            [i0, i1, i2, i3, i4, i5, i6, i7] => Index::I8(*i0, *i1, *i2, *i3, *i4, *i5, *i6, *i7),
            [i0, i1, i2, i3, i4, i5, i6, i7, i8] => {
                Index::I9(*i0, *i1, *i2, *i3, *i4, *i5, *i6, *i7, *i8)
            }
            [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9] => {
                Index::I10(*i0, *i1, *i2, *i3, *i4, *i5, *i6, *i7, *i8, *i9)
            }
            [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10] => {
                Index::I11(*i0, *i1, *i2, *i3, *i4, *i5, *i6, *i7, *i8, *i9, *i10)
            }

            _ => panic!("building indices for higher dimensions not implemented yet"),
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        match &self {
            Index::I1(..) => 1,
            Index::I2(..) => 2,
            Index::I3(..) => 3,
            Index::I4(..) => 4,
            Index::I5(..) => 5,
            Index::I6(..) => 6,
            Index::I7(..) => 7,
            Index::I8(..) => 8,
            Index::I9(..) => 9,
            Index::I10(..) => 10,
            Index::I11(..) => 11,
        }
    }

    pub fn to_vec(&self) -> Vec<usize> {
        match &self {
            Index::I1(i0) => vec![*i0],
            Index::I2(i0, i1) => vec![*i0, *i1],
            Index::I3(i0, i1, i2) => vec![*i0, *i1, *i2],
            Index::I4(i0, i1, i2, i3) => vec![*i0, *i1, *i2, *i3],
            Index::I5(i0, i1, i2, i3, i4) => vec![*i0, *i1, *i2, *i3, *i4],
            Index::I6(i0, i1, i2, i3, i4, i5) => vec![*i0, *i1, *i2, *i3, *i4, *i5],
            Index::I7(i0, i1, i2, i3, i4, i5, i6) => vec![*i0, *i1, *i2, *i3, *i4, *i5, *i6],
            Index::I8(i0, i1, i2, i3, i4, i5, i6, i7) => {
                vec![*i0, *i1, *i2, *i3, *i4, *i5, *i6, *i7]
            }
            Index::I9(i0, i1, i2, i3, i4, i5, i6, i7, i8) => {
                vec![*i0, *i1, *i2, *i3, *i4, *i5, *i6, *i7, *i8]
            }
            Index::I10(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9) => {
                vec![*i0, *i1, *i2, *i3, *i4, *i5, *i6, *i7, *i8, *i9]
            }
            Index::I11(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10) => {
                vec![*i0, *i1, *i2, *i3, *i4, *i5, *i6, *i7, *i8, *i9, *i10]
            }
        }
    }
}

impl Dimension {
    pub fn new(dimensions: &[usize]) -> Dimension {
        match dimensions {
            [n0] => Dimension::D1 { n0: *n0 },
            [n0, n1] => Dimension::D2 { n0: *n0, n1: *n1 },
            [n0, n1, n2] => Dimension::D3 {
                n0: *n0,
                n1: *n1,
                n2: *n2,
                _n1n2: n1 * n2,
            },
            [n0, n1, n2, n3] => Dimension::D4 {
                n0: *n0,
                n1: *n1,
                n2: *n2,
                n3: *n3,
                _n1n2n3: n1 * n2 * n3,
                _n2n3: n2 * n3,
            },
            [n0, n1, n2, n3, n4] => Dimension::D5 {
                n0: *n0,
                n1: *n1,
                n2: *n2,
                n3: *n3,
                n4: *n4,
                _n1n2n3n4: n1 * n2 * n3 * n4,
                _n2n3n4: n2 * n3 * n4,
                _n3n4: n3 * n4,
            },
            [n0, n1, n2, n3, n4, n5] => Dimension::D6 {
                n0: *n0,
                n1: *n1,
                n2: *n2,
                n3: *n3,
                n4: *n4,
                n5: *n5,
                _n1n2n3n4n5: n1 * n2 * n3 * n4 * n5,
                _n2n3n4n5: n2 * n3 * n4 * n5,
                _n3n4n5: n3 * n4 * n5,
                _n4n5: n4 * n5,
            },
            [n0, n1, n2, n3, n4, n5, n6] => Dimension::D7 {
                n0: *n0,
                n1: *n1,
                n2: *n2,
                n3: *n3,
                n4: *n4,
                n5: *n5,
                n6: *n6,
                _n1n2n3n4n5n6: n1 * n2 * n3 * n4 * n5 * n6,
                _n2n3n4n5n6: n2 * n3 * n4 * n5 * n6,
                _n3n4n5n6: n3 * n4 * n5 * n6,
                _n4n5n6: n4 * n5 * n6,
                _n5n6: n5 * n6,
            },
            [n0, n1, n2, n3, n4, n5, n6, n7] => Dimension::D8 {
                n0: *n0,
                n1: *n1,
                n2: *n2,
                n3: *n3,
                n4: *n4,
                n5: *n5,
                n6: *n6,
                n7: *n7,
                _n1n2n3n4n5n6n7: n1 * n2 * n3 * n4 * n5 * n6 * n7,
                _n2n3n4n5n6n7: n2 * n3 * n4 * n5 * n6 * n7,
                _n3n4n5n6n7: n3 * n4 * n5 * n6 * n7,
                _n4n5n6n7: n4 * n5 * n6 * n7,
                _n5n6n7: n5 * n6 * n7,
                _n6n7: n6 * n7,
            },
            [n0, n1, n2, n3, n4, n5, n6, n7, n8] => Dimension::D9 {
                n0: *n0,
                n1: *n1,
                n2: *n2,
                n3: *n3,
                n4: *n4,
                n5: *n5,
                n6: *n6,
                n7: *n7,
                n8: *n8,
                _n1n2n3n4n5n6n7n8: n1 * n2 * n3 * n4 * n5 * n6 * n7 * n8,
                _n2n3n4n5n6n7n8: n2 * n3 * n4 * n5 * n6 * n7 * n8,
                _n3n4n5n6n7n8: n3 * n4 * n5 * n6 * n7 * n8,
                _n4n5n6n7n8: n4 * n5 * n6 * n7 * n8,
                _n5n6n7n8: n5 * n6 * n7 * n8,
                _n6n7n8: n6 * n7 * n8,
                _n7n8: n7 * n8,
            },
            [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9] => Dimension::D10 {
                n0: *n0,
                n1: *n1,
                n2: *n2,
                n3: *n3,
                n4: *n4,
                n5: *n5,
                n6: *n6,
                n7: *n7,
                n8: *n8,
                n9: *n9,
                _n1n2n3n4n5n6n7n8n9: n1 * n2 * n3 * n4 * n5 * n6 * n7 * n8 * n9,
                _n2n3n4n5n6n7n8n9: n2 * n3 * n4 * n5 * n6 * n7 * n8 * n9,
                _n3n4n5n6n7n8n9: n3 * n4 * n5 * n6 * n7 * n8 * n9,
                _n4n5n6n7n8n9: n4 * n5 * n6 * n7 * n8 * n9,
                _n5n6n7n8n9: n5 * n6 * n7 * n8 * n9,
                _n6n7n8n9: n6 * n7 * n8 * n9,
                _n7n8n9: n7 * n8 * n9,
                _n8n9: n8 * n9,
            },
            [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10] => Dimension::D11 {
                n0: *n0,
                n1: *n1,
                n2: *n2,
                n3: *n3,
                n4: *n4,
                n5: *n5,
                n6: *n6,
                n7: *n7,
                n8: *n8,
                n9: *n9,
                n10: *n10,
                _n1n2n3n4n5n6n7n8n9n10: n1 * n2 * n3 * n4 * n5 * n6 * n7 * n8 * n9 * n10,
                _n2n3n4n5n6n7n8n9n10: n2 * n3 * n4 * n5 * n6 * n7 * n8 * n9 * n10,
                _n3n4n5n6n7n8n9n10: n3 * n4 * n5 * n6 * n7 * n8 * n9 * n10,
                _n4n5n6n7n8n9n10: n4 * n5 * n6 * n7 * n8 * n9 * n10,
                _n5n6n7n8n9n10: n5 * n6 * n7 * n8 * n9 * n10,
                _n6n7n8n9n10: n6 * n7 * n8 * n9 * n10,
                _n7n8n9n10: n7 * n8 * n9 * n10,
                _n8n9n10: n8 * n9 * n10,
                _n9n10: n9 * n10,
            },
            _ => panic!("building higher dimensions is not implemented yet"),
        }
    }
}

// TODO [nku] [later] index function should be tested
fn index(idx: &Index, dim: &Dimension) -> usize {
    // idx.0 * (N1*N2*N3*...) + idx.1 * (N2*N3*...) + ... + idx.k
    match (idx, dim) {
        (Index::I1(i0), Dimension::D1 { n0: _ }) => *i0,
        (Index::I2(i0, i1), Dimension::D2 { n0: _, n1 }) => i0 * n1 + i1,
        (
            Index::I3(i0, i1, i2),
            Dimension::D3 {
                n0: _,
                n1: _,
                n2,
                _n1n2,
            },
        ) => i0 * _n1n2 + i1 * n2 + i2,
        (
            Index::I4(i0, i1, i2, i3),
            Dimension::D4 {
                n0: _,
                n1: _,
                n2: _,
                n3,
                _n1n2n3,
                _n2n3,
            },
        ) => i0 * _n1n2n3 + i1 * _n2n3 + i2 * n3 + i3,
        (
            Index::I5(i0, i1, i2, i3, i4),
            Dimension::D5 {
                n0: _,
                n1: _,
                n2: _,
                n3: _,
                n4,
                _n1n2n3n4,
                _n2n3n4,
                _n3n4,
            },
        ) => i0 * _n1n2n3n4 + i1 * _n2n3n4 + i2 * _n3n4 + i3 * n4 + i4,
        (
            Index::I6(i0, i1, i2, i3, i4, i5),
            Dimension::D6 {
                n0: _,
                n1: _,
                n2: _,
                n3: _,
                n4: _,
                n5,
                _n1n2n3n4n5,
                _n2n3n4n5,
                _n3n4n5,
                _n4n5,
            },
        ) => i0 * _n1n2n3n4n5 + i1 * _n2n3n4n5 + i2 * _n3n4n5 + i3 * _n4n5 + i4 * n5 + i5,
        (
            Index::I7(i0, i1, i2, i3, i4, i5, i6),
            Dimension::D7 {
                n0: _,
                n1: _,
                n2: _,
                n3: _,
                n4: _,
                n5: _,
                n6,
                _n1n2n3n4n5n6,
                _n2n3n4n5n6,
                _n3n4n5n6,
                _n4n5n6,
                _n5n6,
            },
        ) => {
            i0 * _n1n2n3n4n5n6
                + i1 * _n2n3n4n5n6
                + i2 * _n3n4n5n6
                + i3 * _n4n5n6
                + i4 * _n5n6
                + i5 * n6
                + i6
        }
        (
            Index::I8(i0, i1, i2, i3, i4, i5, i6, i7),
            Dimension::D8 {
                n0: _,
                n1: _,
                n2: _,
                n3: _,
                n4: _,
                n5: _,
                n6: _,
                n7,
                _n1n2n3n4n5n6n7,
                _n2n3n4n5n6n7,
                _n3n4n5n6n7,
                _n4n5n6n7,
                _n5n6n7,
                _n6n7,
            },
        ) => {
            i0 * _n1n2n3n4n5n6n7
                + i1 * _n2n3n4n5n6n7
                + i2 * _n3n4n5n6n7
                + i3 * _n4n5n6n7
                + i4 * _n5n6n7
                + i5 * _n6n7
                + i6 * n7
                + i7
        }
        (
            Index::I9(i0, i1, i2, i3, i4, i5, i6, i7, i8),
            Dimension::D9 {
                n0: _,
                n1: _,
                n2: _,
                n3: _,
                n4: _,
                n5: _,
                n6: _,
                n7: _,
                n8,
                _n1n2n3n4n5n6n7n8,
                _n2n3n4n5n6n7n8,
                _n3n4n5n6n7n8,
                _n4n5n6n7n8,
                _n5n6n7n8,
                _n6n7n8,
                _n7n8,
            },
        ) => {
            i0 * _n1n2n3n4n5n6n7n8
                + i1 * _n2n3n4n5n6n7n8
                + i2 * _n3n4n5n6n7n8
                + i3 * _n4n5n6n7n8
                + i4 * _n5n6n7n8
                + i5 * _n6n7n8
                + i6 * _n7n8
                + i7 * n8
                + i8
        }
        (
            Index::I10(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9),
            Dimension::D10 {
                n0: _,
                n1: _,
                n2: _,
                n3: _,
                n4: _,
                n5: _,
                n6: _,
                n7: _,
                n8: _,
                n9,
                _n1n2n3n4n5n6n7n8n9,
                _n2n3n4n5n6n7n8n9,
                _n3n4n5n6n7n8n9,
                _n4n5n6n7n8n9,
                _n5n6n7n8n9,
                _n6n7n8n9,
                _n7n8n9,
                _n8n9,
            },
        ) => {
            i0 * _n1n2n3n4n5n6n7n8n9
                + i1 * _n2n3n4n5n6n7n8n9
                + i2 * _n3n4n5n6n7n8n9
                + i3 * _n4n5n6n7n8n9
                + i4 * _n5n6n7n8n9
                + i5 * _n6n7n8n9
                + i6 * _n7n8n9
                + i7 * _n8n9
                + i8 * n9
                + i9
        }
        (
            Index::I11(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10),
            Dimension::D11 {
                n0: _,
                n1: _,
                n2: _,
                n3: _,
                n4: _,
                n5: _,
                n6: _,
                n7: _,
                n8: _,
                n9: _,
                n10,
                _n1n2n3n4n5n6n7n8n9n10,
                _n2n3n4n5n6n7n8n9n10,
                _n3n4n5n6n7n8n9n10,
                _n4n5n6n7n8n9n10,
                _n5n6n7n8n9n10,
                _n6n7n8n9n10,
                _n7n8n9n10,
                _n8n9n10,
                _n9n10,
            },
        ) => {
            i0 * _n1n2n3n4n5n6n7n8n9n10
                + i1 * _n2n3n4n5n6n7n8n9n10
                + i2 * _n3n4n5n6n7n8n9n10
                + i3 * _n4n5n6n7n8n9n10
                + i4 * _n5n6n7n8n9n10
                + i5 * _n6n7n8n9n10
                + i6 * _n7n8n9n10
                + i7 * _n8n9n10
                + i8 * _n9n10
                + i9 * n10
                + i10
        }
        _ => panic!("index and dimension must match"),
    }
}

// TODO [nku] [later] test that from_index(index(x)) == x
pub fn from_idx(idx: usize, dim: &Dimension) -> Index {
    let dims = match dim {
        Dimension::D1 { n0 } => vec![n0],
        Dimension::D2 { n0, n1 } => vec![n0, n1],
        Dimension::D3 { n0, n1, n2, .. } => vec![n0, n1, n2],
        Dimension::D4 { n0, n1, n2, n3, .. } => vec![n0, n1, n2, n3],
        Dimension::D5 {
            n0, n1, n2, n3, n4, ..
        } => vec![n0, n1, n2, n3, n4],
        Dimension::D6 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            ..
        } => vec![n0, n1, n2, n3, n4, n5],
        Dimension::D7 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            ..
        } => vec![n0, n1, n2, n3, n4, n5, n6],
        Dimension::D8 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            n7,
            ..
        } => vec![n0, n1, n2, n3, n4, n5, n6, n7],
        Dimension::D9 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            n7,
            n8,
            ..
        } => vec![n0, n1, n2, n3, n4, n5, n6, n7, n8],
        Dimension::D10 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            n7,
            n8,
            n9,
            ..
        } => vec![n0, n1, n2, n3, n4, n5, n6, n7, n8, n9],
        Dimension::D11 {
            n0,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            n7,
            n8,
            n9,
            n10,
            ..
        } => vec![n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10],
    };

    let mut nidx = Vec::new();
    let mut idx = idx;
    for dim in dims.into_iter().rev() {
        let d = idx % dim;
        idx = (idx - d) / dim;
        nidx.push(d);
    }

    match nidx.len() {
        1 => Index::I1(nidx[0]),
        2 => Index::I2(nidx[1], nidx[0]),
        3 => Index::I3(nidx[2], nidx[1], nidx[0]),
        4 => Index::I4(nidx[3], nidx[2], nidx[1], nidx[0]),
        5 => Index::I5(nidx[4], nidx[3], nidx[2], nidx[1], nidx[0]),
        6 => Index::I6(nidx[5], nidx[4], nidx[3], nidx[2], nidx[1], nidx[0]),
        7 => Index::I7(
            nidx[6], nidx[5], nidx[4], nidx[3], nidx[2], nidx[1], nidx[0],
        ),
        8 => Index::I8(
            nidx[7], nidx[6], nidx[5], nidx[4], nidx[3], nidx[2], nidx[1], nidx[0],
        ),
        9 => Index::I9(
            nidx[8], nidx[7], nidx[6], nidx[5], nidx[4], nidx[3], nidx[2], nidx[1], nidx[0],
        ),
        10 => Index::I10(
            nidx[9], nidx[8], nidx[7], nidx[6], nidx[5], nidx[4], nidx[3], nidx[2], nidx[1],
            nidx[0],
        ),
        11 => Index::I11(
            nidx[10], nidx[9], nidx[8], nidx[7], nidx[6], nidx[5], nidx[4], nidx[3], nidx[2],
            nidx[1], nidx[0],
        ),
        _ => panic!("not implemented yet"),
    }
}

impl<T: Clone> NArray<T> {
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.array.iter()
    }

    pub fn get_by_flat(&self, idx: usize) -> &T {
        &self.array[idx]
    }

    #[allow(dead_code)]
    pub fn get(&self, idx: &Index) -> &T {
        let idx = index(idx, &self.dim);
        &self.array[idx]
    }

    #[allow(dead_code)]
    pub fn set(&mut self, idx: &Index, new_value: T) {
        let idx = index(idx, &self.dim);
        self.array[idx] = new_value;
    }

    pub fn update<F, K>(&mut self, idx: &Index, input: &K, func: F)
    where
        F: Fn(&mut T, &K),
    {
        let idx = index(idx, &self.dim);
        func(&mut self.array[idx], input);
    }
}
