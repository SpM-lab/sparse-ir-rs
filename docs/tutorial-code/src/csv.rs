//! The smallest CSV reader and writer that the tutorial needs.
//!
//! The files have one comment line, then a header of column names, then rows
//! of numbers. Numbers are written with Rust's shortest representation that
//! parses back to the same `f64`, so writing and reading a table never loses a
//! bit, and two runs that computed the same numbers produce byte-identical
//! files.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;

/// A table of named columns of equal length.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    /// The `#` line at the top of the file, without the leading `#`.
    pub comment: String,
    pub columns: Vec<(String, Vec<f64>)>,
}

impl Table {
    /// A table whose comment records which example and library version wrote
    /// it.
    pub fn new(comment: impl Into<String>) -> Self {
        Self {
            comment: comment.into(),
            columns: Vec::new(),
        }
    }

    /// Appends a column. Panics if its length differs from the columns that
    /// are already there, which would mean the example lined up the wrong
    /// arrays.
    pub fn push(&mut self, name: impl Into<String>, values: impl Into<Vec<f64>>) -> &mut Self {
        let name = name.into();
        let values = values.into();
        if let Some((first_name, first)) = self.columns.first() {
            assert_eq!(
                values.len(),
                first.len(),
                "column `{name}` has {} rows but `{first_name}` has {}",
                values.len(),
                first.len()
            );
        }
        self.columns.push((name, values));
        self
    }

    /// The column with this name.
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    /// The column with this name, or a panic naming the columns that are
    /// there — which is what a verification test wants when a file turns out
    /// to have a different shape than expected.
    pub fn expect_column(&self, name: &str) -> &[f64] {
        self.column(name).unwrap_or_else(|| {
            let names: Vec<&str> = self.columns.iter().map(|(n, _)| n.as_str()).collect();
            panic!("no column `{name}`; the table has {names:?}")
        })
    }

    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, |(_, v)| v.len())
    }

    /// The file's text, as [`write_table`] would write it.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        if self.comment.starts_with('#') {
            out.push_str(&self.comment);
        } else {
            out.push('#');
            out.push(' ');
            out.push_str(&self.comment);
        }
        out.push('\n');

        let names: Vec<&str> = self.columns.iter().map(|(n, _)| n.as_str()).collect();
        out.push_str(&names.join(","));
        out.push('\n');

        for row in 0..self.rows() {
            for (index, (_, values)) in self.columns.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                // `{:e}` is the shortest form that parses back bit-for-bit.
                write!(out, "{:e}", values[row]).expect("writing to a String cannot fail");
            }
            out.push('\n');
        }
        out
    }
}

/// Writes the table to `path`, creating the parent directory if needed.
pub fn write_table(path: &Path, table: &Table) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, table.to_csv())
}

/// Reads a table written by [`write_table`], or by
/// `docs/tutorial-code/scripts/make_reference.py`.
pub fn read_table(path: &Path) -> io::Result<Table> {
    let text = fs::read_to_string(path)?;
    parse_table(&text).map_err(|message| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {message}", path.display()),
        )
    })
}

fn parse_table(text: &str) -> Result<Table, String> {
    let mut lines = text.lines();
    let comment = lines
        .next()
        .ok_or("the file is empty")?
        .trim_start_matches('#')
        .trim()
        .to_string();
    let header = lines.next().ok_or("the file has no header line")?;

    let mut table = Table::new(comment);
    let names: Vec<String> = header.split(',').map(|n| n.trim().to_string()).collect();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); names.len()];

    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != names.len() {
            return Err(format!(
                "row {} has {} fields but the header has {}",
                index + 1,
                fields.len(),
                names.len()
            ));
        }
        for (column, field) in columns.iter_mut().zip(fields) {
            let value = field
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("row {}: `{}`: {e}", index + 1, field.trim()))?;
            column.push(value);
        }
    }

    for (name, values) in names.into_iter().zip(columns) {
        table.push(name, values);
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_survives_a_round_trip_bit_for_bit() {
        let mut table = Table::new("example=demo sparse-ir=0.0.0");
        table.push("x", vec![0.0, -1.0, 1.0 / 3.0, 1e-300, 1e300]);
        table.push("y", vec![f64::MIN_POSITIVE, 2.5, -0.1, 12345.678, 7.0]);

        let text = table.to_csv();
        let back = parse_table(&text).expect("the text we just wrote must parse");
        assert_eq!(back, table);
        assert_eq!(back.to_csv(), text);
    }

    #[test]
    fn a_short_row_is_reported_rather_than_silently_padded() {
        let text = "# c\na,b\n1,2\n3\n";
        let error = parse_table(text).expect_err("the second row is short");
        assert!(error.contains("row 2"), "{error}");
    }
}
