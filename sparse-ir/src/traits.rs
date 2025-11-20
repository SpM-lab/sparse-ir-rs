//! Common trait definitions for SparseIR

/// Statistics type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Statistics {
    Fermionic,
    Bosonic,
}

/// Statistics type trait for compile-time type-level distinction
pub trait StatisticsType: Copy {
    const STATISTICS: Statistics;
}

/// Fermionic statistics marker type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fermionic;

impl StatisticsType for Fermionic {
    const STATISTICS: Statistics = Statistics::Fermionic;
}

/// Bosonic statistics marker type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bosonic;

impl StatisticsType for Bosonic {
    const STATISTICS: Statistics = Statistics::Bosonic;
}

/// Trait for types that can represent statistics at runtime
pub trait StatisticsMarker {
    fn statistics() -> Statistics;
}

impl StatisticsMarker for Fermionic {
    fn statistics() -> Statistics {
        Statistics::Fermionic
    }
}

impl StatisticsMarker for Bosonic {
    fn statistics() -> Statistics {
        Statistics::Bosonic
    }
}

/// Utility functions for statistics
impl Statistics {
    /// Check if this statistics type is fermionic
    pub fn is_fermionic(self) -> bool {
        matches!(self, Statistics::Fermionic)
    }

    /// Check if this statistics type is bosonic
    pub fn is_bosonic(self) -> bool {
        matches!(self, Statistics::Bosonic)
    }

    /// Get the string representation of the statistics
    pub fn as_str(self) -> &'static str {
        match self {
            Statistics::Fermionic => "fermionic",
            Statistics::Bosonic => "bosonic",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_type_trait() {
        assert_eq!(Fermionic::STATISTICS, Statistics::Fermionic);
        assert_eq!(Bosonic::STATISTICS, Statistics::Bosonic);
    }

    #[test]
    fn test_statistics_marker_trait() {
        assert_eq!(Fermionic::statistics(), Statistics::Fermionic);
        assert_eq!(Bosonic::statistics(), Statistics::Bosonic);
    }

    #[test]
    fn test_statistics_utility_methods() {
        assert!(Statistics::Fermionic.is_fermionic());
        assert!(!Statistics::Fermionic.is_bosonic());
        assert!(!Statistics::Bosonic.is_fermionic());
        assert!(Statistics::Bosonic.is_bosonic());

        assert_eq!(Statistics::Fermionic.as_str(), "fermionic");
        assert_eq!(Statistics::Bosonic.as_str(), "bosonic");
    }
}
