# CompoSET Compositional Taxonomy

CompoSET tests whether vision-language models can detect fine-grained compositional differences within a fixed scene. The taxonomy below organizes the compositional phenomena the benchmark covers into four categories, plus two cross-cutting operation types (modify and swap).

Each category asks a different question about the scene: *what* (Identity & Appearance), *how many* (Quantity), *where* (Spatial Relations), and *how* (Activity & State).

---

## Categories

### Identity & Appearance (*what*)

Object identity and object-level visual attributes. Each edit changes one attribute of one object, or replaces one object with another.

| Trait | Examples | edit\_type |
|-------|----------|-----------|
| **Color** | red coat -> blue coat, brown cup -> green cup | `modify_color`, `swap_color` |
| **Pattern** | solid -> striped, plain -> checkered, polka dots, plaid | `modify_pattern` |
| **Material** | wood, metal, glass, ceramic, fabric, plastic (e.g., wooden chair -> metal chair) | `modify_material`, `swap_material` |
| **Shape** | round table -> square table, rectangular -> circular | `modify_shape` |
| **Transparency** | opaque, semi-translucent, transparent | `modify_transparency` |
| **Object identity** | newspaper -> book, ceramic bowl -> ceramic plate | `modify_object` |

**Shape vs object replacement:** A shape change keeps the object category (a table remains a table; only its shape changes). An object replacement switches category (newspaper -> book). This boundary distinguishes `modify_shape` from `modify_object`.

### Quantity (*how many*)

How many instances of an object are present.

| Trait | Examples | edit\_type |
|-------|----------|-----------|
| **Cardinality** | one -> two, two -> several, three -> one | `modify_cardinality` |
| **Absence / presence** | vase on table -> no vase; object present vs absent | `modify_presence` |

> **Design note -- Absence / presence:** Included provisionally. Tests whether a model understands "with X" vs "without X," which linguistically involves negation -- a compositional operation. However: (a) it may reduce to object detection rather than compositionality, and (b) unambiguous testing is hard (does "no vase on the table" mean the vase is absent entirely, or just not on the table?). Requires careful caption design.

### Spatial Relations (*where*)

Where objects are relative to each other or to the frame.

| Trait | Examples | edit\_type |
|-------|----------|-----------|
| **Relative position** | left of / right of, above / below, next to, on / in / under / against, hanging from, in front of / behind | `modify_spatial`, `swap_spatial` |

Relative position is a single trait covering all spatial prepositions -- lateral (left/right), vertical (above/below), support/containment (on/in/under/against), and depth/occlusion (in front of/behind). This avoids overlap between preposition types that describe the same underlying phenomenon (spatial arrangement between two objects).

**modify vs swap for spatial:** `modify_spatial` changes the position of one object (or one group moved together) relative to an anchor that stays fixed -- e.g. "a mug on the left side of the board" vs "a mug on the right side of the board". `swap_spatial` exchanges the positions of two named objects -- e.g. "a pot on the left and a tin on the right" vs "a pot on the right and a tin on the left". The distinction parallels `modify_color` / `swap_color` and tests a different compositional primitive: single-object relation change vs paired binding of identity to position.

> **Note on absolute position and relative scale:** earlier drafts included `modify_absolute_position` (left/right/center of frame, foreground/background) and `modify_relative_scale` (cup bigger than bowl vs bowl bigger than cup) as additional spatial traits. Both were dropped before the v1.0 freeze and are not represented in the released benchmark.

### Activity & State (*how*)

What agents are doing, how they are configured, and what condition objects are in. State (open/closed, on/off, etc.) was a separate category in earlier drafts; it is now folded into this one to keep the four-category structure balanced.

| Trait | Examples | edit\_type |
|-------|----------|-----------|
| **Action** | reading, pouring, cutting, waving | `modify_action` |
| **Pose** | sitting/standing, arms raised (agents); upright/inverted, facing left/right, laying flat (objects) | `modify_pose` |
| **State** | open/closed, on/off, intact/broken, full/empty, wet/dry, lit/unlit, folded/unfolded, etc. | `modify_state` |
| **Role assignment** | dog chasing cat vs cat chasing dog (who does what to whom) | `swap_role` |

Pose covers both body configuration for agents and orientation for inanimate objects (upright/inverted, facing direction). Role assignment tests predicate-argument structure -- a classic compositionality challenge. State captures object condition that is not an appearance attribute (not about what it looks like inherently) and not spatial -- a distinct axis of meaning that prior benchmarks largely miss.

---

## Operation types

Operations cross-cut the categories above. They describe *how* an edit is structured, not *what* attribute it targets.

| Operation | Description | Applies to |
|-----------|-------------|------------|
| **modify** | Change one attribute of one object | All categories (most edits) |
| **swap** | Exchange an attribute (or position) between two objects (tests binding) | Identity & Appearance: `swap_color`, `swap_material`; Spatial Relations: `swap_spatial`; Activity & State: `swap_role` |

---

## Excluded

The following are deliberately excluded from CompoSET. They require either human subjects, environment-level rendering, or inherently ambiguous annotations that would undermine the benchmark's controlled single-edit design.

- **Age, gender, emotion** -- person-centric properties; ethically and perceptually complex
- **Absolute size** -- perceptually ambiguous without reference objects; hard to control in generation
- **Aesthetics** -- subjective judgments (e.g., "pretty," "elegant"); not compositionally testable
- **Scene-level properties** -- location, lighting, season, weather (snow/wet), indoor/outdoor; these are environment changes, not object-level compositional edits

## Deferred

- **Ordinal position** ("first," "second," "last in a row") -- deferred from the initial expansion. Ordinal position decomposes into cardinality + spatial position and is not a distinct compositional primitive. It also poses generation challenges: maintaining a clear linear arrangement where "first" vs "second" is unambiguous requires a strict spatial convention (left-to-right? front-to-back?), and image generators often fail at this. Can be revisited as a stress-test category once the generation pipeline is validated.

---

## State ideas

Not formal subcategories, but a pool to draw from when designing scenes. Some are niche -- use judgment.

- **Functional:** open/closed, on/off, locked/unlocked, folded/unfolded
- **Physical integrity:** intact/broken, cracked, bent/straight, torn/whole
- **Fullness:** full/empty/half-full
- **Phase / condition:** wet/dry, clean/dirty, ripe/unripe, cooked/raw, lit/unlit, inflated/deflated
- **Configuration:** assembled/disassembled, stacked/unstacked, zipped/unzipped
