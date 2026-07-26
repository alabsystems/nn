// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Detect SMT proofs that are UNSAT for reasons unrelated to the property.
//!
//! Every proof here asserts hypotheses `H` and the negated property `¬P`, then
//! expects UNSAT. That is only evidence for `P` if `H ∧ ¬P` is unsatisfiable
//! *because of* `H`. When `¬P` is unsatisfiable on its own, the solver's UNSAT
//! says nothing about the network — the proof passes and proves nothing.
//!
//! Three shapes of that mistake are decidable from the query text alone, with no
//! false positives:
//!
//! - [`VacuitySmell::SelfComparison`] — the violation is `(not (= X X))`. Writing
//!   `let att = a.clone(); att.ne(a)` produces exactly this.
//! - [`VacuitySmell::NegatesOwnHypothesis`] — the violation is `(not (= X Y))`
//!   and `(assert (= X Y))` appears earlier, so the query is `P ∧ ¬P`.
//! - [`VacuitySmell::DefinitionalAlias`] — the two sides of the negated equality
//!   are *defined* by the same computation, as in `y2 = x2 + d` and
//!   `y2_after = x2 + d`. This is invisible in the surface text, but it is
//!   recovered by tracking each variable's lineage: substitute every variable's
//!   defining assertion and normalize modulo reordering of `+`/`*`/`and`/`or`. If
//!   both sides collapse to the same *compound* term, their equality holds in
//!   every model and the negation is UNSAT regardless of the inputs.
//!
//! The lineage normalizer is deliberately weak — only definition substitution
//! and associative-commutative reordering, never distributivity or ring
//! identities — so a flagged query is genuinely the same computation written
//! twice, while a proof that needs real solver reasoning (`(a+b)(a-b) = a²-b²`)
//! is never flagged. It also fires only on a shared *compound* normal form, so a
//! plain equality-transitivity chain (`att = at`, `at = a` ⊢ `att = a`), which
//! the suite treats as legitimate derivation, is left alone.
//!
//! A mutation test remains the backstop for the shapes no static check reaches
//! (e.g. a definition threaded through a symbol-to-symbol rename): perturb the
//! model so the property must break, and assert the query turns SAT.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// A reason a query is UNSAT independently of its hypotheses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VacuitySmell {
    /// The violation compares a term to itself: `(not (= X X))`.
    SelfComparison {
        /// The repeated term.
        term: String,
    },
    /// The violation negates an equality that was asserted as a hypothesis.
    NegatesOwnHypothesis {
        /// The equality, rendered canonically.
        equality: String,
    },
    /// The two sides of the negated equality are *defined* by the same
    /// computation: substituting each variable's defining assertion and
    /// normalizing (modulo `+`/`*`/`and`/`or` reordering) collapses both sides to
    /// the same compound term, so the equality holds in every model regardless of
    /// the inputs.
    DefinitionalAlias {
        /// The shared normal form both sides reduce to.
        normal_form: String,
    },
}

impl fmt::Display for VacuitySmell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SelfComparison { term } => {
                write!(
                    f,
                    "violation compares `{term}` to itself: UNSAT regardless of hypotheses"
                )
            }
            Self::NegatesOwnHypothesis { equality } => write!(
                f,
                "violation negates its own hypothesis `{equality}`: the query is P and not-P",
            ),
            Self::DefinitionalAlias { normal_form } => write!(
                f,
                "violation negates an equality whose sides both reduce to `{normal_form}` after \
                 substituting their definitions: UNSAT regardless of the inputs",
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Atom(String),
    List(Vec<Node>),
}

impl Node {
    fn render(&self) -> String {
        match self {
            Self::Atom(a) => a.clone(),
            Self::List(items) => {
                let inner: Vec<String> = items.iter().map(Self::render).collect();
                format!("({})", inner.join(" "))
            }
        }
    }

    fn head(&self) -> Option<&str> {
        match self {
            Self::List(items) => match items.first() {
                Some(Self::Atom(a)) => Some(a.as_str()),
                _ => None,
            },
            Self::Atom(_) => None,
        }
    }

    fn args(&self) -> &[Node] {
        match self {
            Self::List(items) => &items[1..],
            Self::Atom(_) => &[],
        }
    }
}

/// Parse a whole SMT-LIB2 script into top-level forms. Unbalanced input yields
/// whatever parsed cleanly; the caller only inspects `assert` forms.
fn parse_forms(src: &str) -> Vec<Node> {
    let mut chars = src.chars().peekable();
    let mut forms = Vec::new();
    while chars.peek().is_some() {
        skip_trivia(&mut chars);
        if chars.peek() != Some(&'(') {
            if chars.next().is_none() {
                break;
            }
            continue;
        }
        if let Some(node) = parse_node(&mut chars) {
            forms.push(node);
        } else {
            break;
        }
    }
    forms
}

fn skip_trivia(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while let Some(&c) = chars.peek() {
        if c == ';' {
            for c in chars.by_ref() {
                if c == '\n' {
                    break;
                }
            }
        } else if c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
}

fn parse_node(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Option<Node> {
    skip_trivia(chars);
    match chars.peek()? {
        '(' => {
            chars.next();
            let mut items = Vec::new();
            loop {
                skip_trivia(chars);
                match chars.peek()? {
                    ')' => {
                        chars.next();
                        return Some(Node::List(items));
                    }
                    _ => items.push(parse_node(chars)?),
                }
            }
        }
        ')' => None,
        '|' => {
            chars.next();
            let mut atom = String::from("|");
            for c in chars.by_ref() {
                atom.push(c);
                if c == '|' {
                    break;
                }
            }
            Some(Node::Atom(atom))
        }
        _ => {
            let mut atom = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '(' || c == ')' {
                    break;
                }
                atom.push(c);
                chars.next();
            }
            (!atom.is_empty()).then_some(Node::Atom(atom))
        }
    }
}

/// Canonical key for an equality, order-insensitive (`=` is symmetric).
fn equality_key(a: &Node, b: &Node) -> String {
    let (x, y) = (a.render(), b.render());
    if x <= y {
        format!("(= {x} {y})")
    } else {
        format!("(= {y} {x})")
    }
}

/// Cap on lineage substitution depth — guards against cyclic definitions and
/// pathological nesting. Far deeper than any real proof's definition chain.
const MAX_SUBST_DEPTH: usize = 64;

/// The names introduced by `declare-const` or nullary `declare-fun`, i.e. the
/// query's variables. Restricting definition tracking to these keeps operators
/// and numerals from being mistaken for definable symbols.
fn declared_symbols(forms: &[Node]) -> HashSet<String> {
    let mut set = HashSet::new();
    for form in forms {
        match form.head() {
            Some("declare-const") => {
                if let Some(Node::Atom(name)) = form.args().first() {
                    set.insert(name.clone());
                }
            }
            Some("declare-fun") => {
                let args = form.args();
                if let (Some(Node::Atom(name)), Some(Node::List(params))) =
                    (args.first(), args.get(1))
                {
                    if params.is_empty() {
                        set.insert(name.clone());
                    }
                }
            }
            _ => {}
        }
    }
    set
}

/// This node as a declared variable name, if it is one.
fn as_declared_symbol<'a>(node: &'a Node, declared: &HashSet<String>) -> Option<&'a str> {
    match node {
        Node::Atom(a) if declared.contains(a) => Some(a.as_str()),
        _ => None,
    }
}

/// Map each variable to its defining term, from hypotheses of the form
/// `(= sym term)` where exactly one side is a declared variable. A variable
/// carrying more than one such equality is a *constraint*, not a definition, and
/// is dropped — substituting one arbitrarily would be unsound.
fn definition_map<'a>(hypotheses: &[&'a Node], declared: &HashSet<String>) -> HashMap<String, &'a Node> {
    let mut defs: HashMap<String, &Node> = HashMap::new();
    let mut multiply_defined: HashSet<String> = HashSet::new();
    for h in hypotheses {
        if h.head() != Some("=") || h.args().len() != 2 {
            continue;
        }
        let (a, b) = (&h.args()[0], &h.args()[1]);
        let (sym, term) = match (as_declared_symbol(a, declared), as_declared_symbol(b, declared)) {
            (Some(s), None) => (s.to_string(), b),
            (None, Some(s)) => (s.to_string(), a),
            // Both sides variables (a rename) or neither (a plain fact): not a
            // definition we can orient soundly.
            _ => continue,
        };
        if defs.insert(sym.clone(), term).is_some() {
            multiply_defined.insert(sym);
        }
    }
    for k in multiply_defined {
        defs.remove(&k);
    }
    defs
}

/// Substitute definitions recursively, then reorder associative-commutative
/// operators into a canonical form. Two terms that are the same computation up
/// to variable definitions and `+`/`*`/`and`/`or` reordering share a result;
/// nothing stronger (no distributivity, no ring normalization) is applied, so
/// terms that are only *semantically* equal keep distinct forms.
fn normalize(node: &Node, defs: &HashMap<String, &Node>, depth: usize) -> Node {
    if depth == 0 {
        return node.clone();
    }
    match node {
        Node::Atom(a) => match defs.get(a) {
            Some(def) => normalize(def, defs, depth - 1),
            None => node.clone(),
        },
        Node::List(items) => {
            let normed: Vec<Node> = items.iter().map(|n| normalize(n, defs, depth - 1)).collect();
            canonicalize(normed)
        }
    }
}

/// Flatten and sort the operands of AC operators (`+`, `*`, `and`, `or`) and the
/// two operands of symmetric `=`/`distinct`. Non-commutative operators (`-`,
/// `/`, `<`, ...) keep operand order, since reordering them would be unsound.
fn canonicalize(items: Vec<Node>) -> Node {
    let Some(Node::Atom(op)) = items.first().cloned() else {
        return Node::List(items);
    };
    if matches!(op.as_str(), "+" | "*" | "and" | "or") {
        let mut flat: Vec<Node> = Vec::new();
        for arg in items.into_iter().skip(1) {
            match arg {
                Node::List(inner) if inner.first() == Some(&Node::Atom(op.clone())) => {
                    flat.extend(inner.into_iter().skip(1));
                }
                other => flat.push(other),
            }
        }
        flat.sort_by(|a, b| a.render().cmp(&b.render()));
        let mut out = vec![Node::Atom(op)];
        out.extend(flat);
        Node::List(out)
    } else if matches!(op.as_str(), "=" | "distinct") && items.len() == 3 {
        let mut pair = vec![items[1].clone(), items[2].clone()];
        pair.sort_by(|a, b| a.render().cmp(&b.render()));
        Node::List(vec![Node::Atom(op), pair.remove(0), pair.remove(0)])
    } else {
        Node::List(items)
    }
}

/// Split a violation into the disjuncts a counterexample could satisfy.
fn disjuncts(node: &Node) -> Vec<&Node> {
    if node.head() == Some("or") {
        node.args().iter().flat_map(disjuncts).collect()
    } else {
        vec![node]
    }
}

/// Downgrade a "proven" verdict to a failure when the query is vacuous.
///
/// Every module's `execute_and_check` funnels its `(proven, detail)` through
/// this before returning it, so a query that is UNSAT only because it asserts
/// `P ∧ ¬P` (or compares a term to itself) never counts as a proof — the
/// module's own `test_*_proven` fails until the proof states a real theorem.
/// A non-vacuous query is returned unchanged, so genuine proofs are unaffected.
pub(crate) fn reject_if_vacuous(smt2: &str, proven: bool, detail: String) -> (bool, String) {
    match (proven, vacuity_smell(smt2)) {
        (true, Some(smell)) => (false, format!("VACUOUS: {smell}")),
        _ => (proven, detail),
    }
}

/// Inspect the SMT-LIB2 text of a proof query for a vacuous UNSAT.
///
/// The last `assert` is taken to be the negated property, every earlier one a
/// hypothesis — the convention every `prove_*` function in this crate follows.
pub(crate) fn vacuity_smell(smt2: &str) -> Option<VacuitySmell> {
    let forms = parse_forms(smt2);
    let asserts: Vec<&Node> = forms
        .iter()
        .filter(|f| f.head() == Some("assert"))
        .filter_map(|f| f.args().first())
        .collect();
    let (violation, hypotheses) = asserts.split_last()?;

    // Equalities asserted outright, i.e. `(assert (= X Y))`.
    let asserted_equalities: Vec<String> = hypotheses
        .iter()
        .filter(|h| h.head() == Some("=") && h.args().len() == 2)
        .map(|h| equality_key(&h.args()[0], &h.args()[1]))
        .collect();

    // Variable lineage: each variable's defining term, for the alias check.
    let declared = declared_symbols(&forms);
    let defs = definition_map(hypotheses, &declared);

    for disjunct in disjuncts(violation) {
        // `Expr::ne` renders as `(not (= a b))`.
        let Some(eq) = negated_equality(disjunct) else {
            continue;
        };
        let (lhs, rhs) = (&eq.args()[0], &eq.args()[1]);
        if lhs == rhs {
            return Some(VacuitySmell::SelfComparison { term: lhs.render() });
        }
        let key = equality_key(lhs, rhs);
        if asserted_equalities.contains(&key) {
            return Some(VacuitySmell::NegatesOwnHypothesis { equality: key });
        }
        // Lineage: if both sides reduce to the same *compound* computation after
        // substituting their definitions, the equality is entailed and its
        // negation is UNSAT regardless of the inputs. A shared bare atom is plain
        // equality transitivity, not aliasing, so it is not flagged.
        let nf_lhs = normalize(lhs, &defs, MAX_SUBST_DEPTH);
        if matches!(nf_lhs, Node::List(_)) && nf_lhs == normalize(rhs, &defs, MAX_SUBST_DEPTH) {
            return Some(VacuitySmell::DefinitionalAlias {
                normal_form: nf_lhs.render(),
            });
        }
    }
    None
}

/// The `(= a b)` inside `(not (= a b))`, or the binary `(distinct a b)`.
fn negated_equality(node: &Node) -> Option<&Node> {
    match node.head() {
        Some("not") => {
            let inner = node.args().first()?;
            (inner.head() == Some("=") && inner.args().len() == 2).then_some(inner)
        }
        Some("distinct") if node.args().len() == 2 => Some(node),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_downgrades_a_vacuous_proof() {
        let smt2 = "(assert (= a b))\n(assert (not (= a b)))\n(check-sat)";
        let (proven, detail) = reject_if_vacuous(smt2, true, "UNSAT".into());
        assert!(!proven);
        assert!(detail.starts_with("VACUOUS:"), "{detail}");
    }

    #[test]
    fn reject_leaves_a_genuine_proof_untouched() {
        let smt2 = "(assert (>= x 0.0))\n(assert (< x 0.0))\n(check-sat)";
        assert_eq!(
            reject_if_vacuous(smt2, true, "UNSAT".into()),
            (true, "UNSAT".to_string()),
        );
    }

    #[test]
    fn flags_a_term_compared_to_itself() {
        let smt2 =
            "(declare-const a Real)\n(assert (>= a 0.0))\n(assert (not (= a a)))\n(check-sat)";
        assert_eq!(
            vacuity_smell(smt2),
            Some(VacuitySmell::SelfComparison { term: "a".into() }),
        );
    }

    #[test]
    fn flags_a_compound_term_compared_to_itself() {
        let smt2 = "(assert (not (= (+ (* i c) j) (+ (* i c) j))))\n(check-sat)";
        assert!(matches!(
            vacuity_smell(smt2),
            Some(VacuitySmell::SelfComparison { .. }),
        ));
    }

    #[test]
    fn flags_a_violation_that_negates_its_own_hypothesis() {
        let smt2 = "(assert (= tin tout))\n(assert (not (= tin tout)))\n(check-sat)";
        assert_eq!(
            vacuity_smell(smt2),
            Some(VacuitySmell::NegatesOwnHypothesis {
                equality: "(= tin tout)".into()
            }),
        );
    }

    /// `=` is symmetric, so the hypothesis may be written either way round.
    #[test]
    fn hypothesis_match_is_order_insensitive() {
        let smt2 = "(assert (= tout tin))\n(assert (not (= tin tout)))\n(check-sat)";
        assert!(matches!(
            vacuity_smell(smt2),
            Some(VacuitySmell::NegatesOwnHypothesis { .. }),
        ));
    }

    /// A disjunctive violation is vacuous if ANY disjunct is.
    #[test]
    fn flags_a_self_comparison_inside_a_disjunction() {
        let smt2 = "(assert (or (not (= m n)) (not (= k k))))\n(check-sat)";
        assert!(matches!(
            vacuity_smell(smt2),
            Some(VacuitySmell::SelfComparison { .. }),
        ));
    }

    #[test]
    fn accepts_a_proof_that_derives_its_conclusion() {
        // att = at, at = a  =>  att = a. The conclusion is not a hypothesis.
        let smt2 = "(assert (= at a))\n(assert (= att at))\n(assert (not (= att a)))\n(check-sat)";
        assert_eq!(vacuity_smell(smt2), None);
    }

    #[test]
    fn accepts_a_violation_over_an_inequality() {
        let smt2 = "(assert (>= x 0.0))\n(assert (> x 1.0))\n(check-sat)";
        assert_eq!(vacuity_smell(smt2), None);
    }

    #[test]
    fn ignores_comments_and_quoted_symbols() {
        let smt2 = "; a comment (not (= a a))\n(assert (not (= |odd sym| |odd sym|)))\n(check-sat)";
        assert!(matches!(
            vacuity_smell(smt2),
            Some(VacuitySmell::SelfComparison { .. }),
        ));
    }

    // --- lineage / definitional aliasing --------------------------------------

    /// Two variables defined by the same computation, then proven equal: the
    /// negation is UNSAT regardless of the inputs, so the proof proves nothing.
    #[test]
    fn flags_two_variables_defined_by_the_same_term() {
        let smt2 = "\
            (declare-const x2 Real)\n\
            (declare-const d Real)\n\
            (declare-const y2 Real)\n\
            (declare-const y2_after Real)\n\
            (assert (= y2 (+ x2 d)))\n\
            (assert (= y2_after (+ x2 d)))\n\
            (assert (not (= y2 y2_after)))\n\
            (check-sat)";
        assert!(
            matches!(vacuity_smell(smt2), Some(VacuitySmell::DefinitionalAlias { .. })),
            "got {:?}",
            vacuity_smell(smt2),
        );
    }

    /// The two definitions differ only by `+` operand order; AC-normalization
    /// still collapses them, so the aliasing is caught.
    #[test]
    fn flags_aliasing_up_to_commutative_reordering() {
        let smt2 = "\
            (declare-const a Real)\n\
            (declare-const b Real)\n\
            (declare-const p Real)\n\
            (declare-const q Real)\n\
            (assert (= p (+ a b)))\n\
            (assert (= q (+ b a)))\n\
            (assert (not (= p q)))\n\
            (check-sat)";
        assert!(
            matches!(vacuity_smell(smt2), Some(VacuitySmell::DefinitionalAlias { .. })),
            "got {:?}",
            vacuity_smell(smt2),
        );
    }

    /// Aliasing threaded one level deep through another defined variable is still
    /// recovered by recursive substitution.
    #[test]
    fn flags_aliasing_through_a_nested_definition() {
        let smt2 = "\
            (declare-const x Real)\n\
            (declare-const w Real)\n\
            (declare-const t Real)\n\
            (declare-const lhs Real)\n\
            (declare-const rhs Real)\n\
            (assert (= t (* w x)))\n\
            (assert (= lhs (+ t 1)))\n\
            (assert (= rhs (+ (* w x) 1)))\n\
            (assert (not (= lhs rhs)))\n\
            (check-sat)";
        assert!(
            matches!(vacuity_smell(smt2), Some(VacuitySmell::DefinitionalAlias { .. })),
            "got {:?}",
            vacuity_smell(smt2),
        );
    }

    /// A plain equality-transitivity chain shares only a bare-atom normal form,
    /// which the suite treats as legitimate derivation — it must NOT be flagged.
    #[test]
    fn does_not_flag_plain_equality_transitivity() {
        let smt2 = "\
            (declare-const a Real)\n\
            (declare-const at Real)\n\
            (declare-const att Real)\n\
            (assert (= at a))\n\
            (assert (= att at))\n\
            (assert (not (= att a)))\n\
            (check-sat)";
        assert_eq!(vacuity_smell(smt2), None);
    }

    /// An algebraic identity that needs distributivity — `(a+b)(a-b) = a²-b²` —
    /// is genuinely solver-proven; the weak normalizer must NOT flag it, or the
    /// gate would reject real theorems.
    #[test]
    fn does_not_flag_a_genuine_algebraic_identity() {
        let smt2 = "\
            (declare-const a Real)\n\
            (declare-const b Real)\n\
            (declare-const lhs Real)\n\
            (declare-const rhs Real)\n\
            (assert (= lhs (* (+ a b) (- a b))))\n\
            (assert (= rhs (- (* a a) (* b b))))\n\
            (assert (not (= lhs rhs)))\n\
            (check-sat)";
        assert_eq!(vacuity_smell(smt2), None);
    }

    /// Two variables defined by *different* computations are not aliases; proving
    /// them equal is a real (here, false) query and must not be flagged.
    #[test]
    fn does_not_flag_distinct_computations() {
        let smt2 = "\
            (declare-const x Real)\n\
            (declare-const p Real)\n\
            (declare-const q Real)\n\
            (assert (= p (+ x 1)))\n\
            (assert (= q (+ x 2)))\n\
            (assert (not (= p q)))\n\
            (check-sat)";
        assert_eq!(vacuity_smell(smt2), None);
    }

    /// A variable carrying two different definitions is a constraint, not a
    /// definition; it must not be substituted (that would be unsound), so a
    /// same-form coincidence built on it is not flagged.
    #[test]
    fn does_not_substitute_a_multiply_defined_variable() {
        let smt2 = "\
            (declare-const x Real)\n\
            (declare-const y Real)\n\
            (declare-const p Real)\n\
            (assert (= p (+ x 1)))\n\
            (assert (= p (* y 2)))\n\
            (assert (not (= p (+ x 1))))\n\
            (check-sat)";
        // `p` is multiply-defined, so it is not substituted; `(= p (+ x 1))` is
        // still an asserted equality, caught as NegatesOwnHypothesis (not alias).
        assert!(matches!(
            vacuity_smell(smt2),
            Some(VacuitySmell::NegatesOwnHypothesis { .. }),
        ));
    }
}
