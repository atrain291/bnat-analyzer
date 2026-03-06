"""Reference catalog of Bharatanatyam adavus extracted from instructional texts.

This module provides structured knowledge about adavu forms, beat patterns,
and key pose descriptions. Used to enrich LLM coaching feedback with
technique-specific reference data.
"""

ADAVU_CATALOG = {
    "tattadavu": {
        "description": (
            "Tattadavu is the most fundamental adavu in Bharatanatyam, focused on "
            "rhythmic foot striking in aramandi (half-sitting) position. The dancer "
            "maintains hands on the waist (natyarambhe) while striking feet alternately. "
            "There are 8 tattadavus, progressing in rhythmic complexity."
        ),
        "variations": {
            "1st": {
                "talam": "Aadhi Taalam (8 beats)",
                "syllables": "Tai Ya Tai Hi",
                "speeds": {
                    "1st": "Tai, Ya, Tai, Hi, Tai, Ya, Tai, Hi",
                },
                "key_poses": [
                    "Strike right foot flat in aramandi, hands on waist",
                    "Strike left foot flat in aramandi, hands on waist",
                ],
                "technique_points": [
                    "The most basic adavu - focus on clean, flat foot strikes",
                    "Aramandi must be established before starting and held throughout",
                    "Alternating right-left strikes should be perfectly even",
                    "Hands on waist (natyarambhe), elbows pointing outward",
                    "Torso completely upright, no bouncing or swaying",
                    "Each strike should produce a clear, audible sound",
                ],
            },
            "2nd": {
                "talam": "Aadhi Taalam (8 beats)",
                "syllables": "Tai Ya Tai Hi",
                "speeds": {
                    "1st": "Tai, Ya, Tai, Hi, Tai, Ya, Tai, Hi",
                },
                "key_poses": [
                    "Strike right foot with slight heel lift, hands on waist",
                    "Strike left foot with slight heel lift, hands on waist",
                ],
                "technique_points": [
                    "Similar to 1st but with a slight heel-lift variation on strikes",
                    "The heel lift is subtle - foot returns flat immediately",
                    "Maintain aramandi depth even with the heel variation",
                    "Rhythm must remain steady and even between strikes",
                ],
            },
            "3rd": {
                "talam": "Aadhi Taalam (8 beats)",
                "syllables": "Tai Tai Tam - Tai Tai Tam",
                "speeds": {
                    "1st": "Tai, Tai, Tam, Pause, Tai, Tai, Tam, Pause",
                    "2nd": "Tai Tai, Tam Pause, Tai Tai, Tam Pause, Tai Tai, Tam Pause, Tai Tai, Tam Pause",
                    "3rd": "Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai / Tam Pause, Tam Pause, Tam Pause, Tam Pause, Tam Pause, Tam Pause, Tam Pause, Tam Pause",
                },
                "key_poses": [
                    "Strike right foot flat in aramandi, hands on waist",
                    "Strike left foot flat in aramandi, hands on waist",
                ],
                "notes": "As the speed increases, the pause decreases correspondingly.",
                "technique_points": [
                    "Maintain consistent aramandi depth throughout all strikes",
                    "Feet should strike flat on the ground, not on toes",
                    "Hands remain firmly on waist (natyarambhe position)",
                    "Torso stays upright and centered, no leaning",
                    "Weight transfers evenly between left and right",
                    "The Tam strike should be slightly more emphatic than Tai",
                ],
            },
            "4th": {
                "talam": "Aadhi Taalam (8 beats)",
                "syllables": "Tai Ya Tai Hi / Tai Ya Tam",
                "speeds": {
                    "1st": "Tai, Ya, Tai, Hi, Tai, Ya, Tam, Pause",
                },
                "key_poses": [
                    "Strike right foot in aramandi, hands on waist",
                    "Strike left foot in aramandi, hands on waist",
                    "Final Tam: emphatic strike with held position",
                ],
                "technique_points": [
                    "Combines patterns from 1st and 3rd tattadavus",
                    "The concluding Tam should be a strong, definitive strike",
                    "Pause after Tam must be a held aramandi, not relaxation",
                    "Transition between Tai-Ya and Tai-Hi patterns must be seamless",
                ],
            },
            "5th": {
                "talam": "Aadhi Taalam (8 beats)",
                "syllables": "Tai Tai Tai Tai Tam",
                "speeds": {
                    "1st": "Tai, Tai, Tai Tai, Tam, Tai, Tai Tai, Tam",
                    "2nd": "Tai, Tam, Tai Tai, Tai Tai, Tam, Tai Tai, Tai Tai, Tam",
                    "3rd": "Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai / Tam, Tam, Tam, Tam, Tam, Tam, Tam, Tam",
                },
                "key_poses": [
                    "Strike right foot repeatedly in aramandi, hands on waist",
                    "Strike left foot repeatedly in aramandi, hands on waist",
                ],
                "notes": (
                    "In the 1st speed, while the first two beats are struck, the 3rd "
                    "and 4th beats 'Tai Tai' are struck in 2nd speed. Speed increases "
                    "correspondingly in 2nd and 3rd speeds."
                ),
                "technique_points": [
                    "Speed transitions should be smooth, not abrupt",
                    "Aramandi depth must not decrease as speed increases",
                    "Right side strikes complete first, then left side mirrors",
                    "Maintain rhythmic precision even at higher speeds",
                ],
            },
            "6th": {
                "talam": "Aadhi Taalam (8 beats)",
                "syllables": "Tai Ya Tai Hi Tai Tai Tam",
                "speeds": {
                    "1st": "Tai, Ya, Tai, Hi, Tai, Tai, Tam, Pause",
                },
                "key_poses": [
                    "Strike right foot in aramandi (Tai Ya Tai Hi pattern)",
                    "Two quick strikes followed by emphatic Tam",
                    "Mirror on left side",
                ],
                "technique_points": [
                    "Combines the basic Tai-Ya pattern with double-strike and Tam",
                    "The two quick Tai-Tai strikes require extra rhythmic precision",
                    "Tam concludes each phrase emphatically",
                    "Aramandi must stay consistent through the acceleration",
                ],
            },
            "7th": {
                "talam": "Aadhi Taalam (8 beats)",
                "syllables": "Tai Tai Tat Tat Tai Tai Tam",
                "speeds": {
                    "1st": "Tai, Tai, Tat, Tat, Tai, Tam, Pause",
                    "2nd": "Tai Tat, Tai Tai, Tam Tai, Tat Tai, Tai Tam, Tai Pause",
                    "3rd": "Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai, Tai Tai / Tat Tat, Tam-, Tat Tat, Tam-, Tat Tat, Tam-, Tat Tat, Tam-",
                },
                "key_poses": [
                    "Strike right foot in aramandi, hands on waist (Tai beats)",
                    "Strike right with slight variation (Tat beats)",
                    "Strike left foot mirroring right (second half)",
                    "Pause position: hold aramandi, 8th count",
                ],
                "technique_points": [
                    "Distinguish clearly between Tai (full strike) and Tat (lighter strike)",
                    "The 8th count pause should be a held aramandi, not a collapse",
                    "Left-right alternation must be symmetrical",
                    "Weight stays centered through the entire sequence",
                ],
            },
            "8th": {
                "talam": "Aadhi Taalam (8 beats)",
                "syllables": "Tai Ya Tai Hi Tai Tai Tat Tat Tai Tai Tam",
                "speeds": {
                    "1st": "Tai, Ya, Tai, Hi, Tai Tai, Tat Tat, Tai Tai, Tam",
                },
                "key_poses": [
                    "Combines elements of all previous tattadavus",
                    "Starts with basic Tai-Ya pattern",
                    "Progresses through Tat-Tat variation",
                    "Concludes with emphatic Tam",
                ],
                "technique_points": [
                    "The most complex tattadavu - demands mastery of all previous patterns",
                    "Smooth transitions between different rhythmic patterns within one phrase",
                    "Aramandi consistency is especially challenging with the complexity",
                    "Each section (Tai-Ya, Tat-Tat, Tam) should be clearly articulated",
                    "The culminating Tam must be strong and grounded",
                ],
            },
        },
    },
    "nattadavu": {
        "description": (
            "Nattadavu involves lateral stretching movements with arms fully extended "
            "to the sides while maintaining aramandi. The dancer extends and strikes "
            "legs outward with coordinated arm movements and wrist turns. "
            "There are 8 nattadavus, building from simple extensions to complex "
            "cross-body coordinations."
        ),
        "variations": {
            "1st": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Yum Tat Ta / Tai Yum Tam",
                "key_poses": [
                    "Aramandi with arms extended horizontally to both sides",
                    "Extend right leg outward to the right side",
                    "Strike right foot back to aramandi",
                    "Repeat on left side, mirroring exactly",
                ],
                "technique_points": [
                    "The foundational nattadavu - establishes lateral extension pattern",
                    "Arms must be perfectly horizontal at shoulder height throughout",
                    "Leg extends outward with pointed toes before striking back",
                    "Aramandi depth maintained even during leg extension",
                    "Torso stays centered, does not lean toward the extending leg",
                    "Both sides must be performed with identical extension and timing",
                ],
            },
            "2nd": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Yum Tat Ta / Tai Yum Tam",
                "key_poses": [
                    "Aramandi with arms extended horizontally",
                    "Extend right leg outward with slight hop",
                    "Strike right foot with wrist turn",
                    "Mirror on left side",
                ],
                "technique_points": [
                    "Adds a subtle hop/spring to the basic nattadavu pattern",
                    "The hop should be controlled, not bouncy - maintain head level",
                    "Wrist turns (rechaka) begin to coordinate with leg movements",
                    "Landing from the hop must return to proper aramandi depth",
                    "Arms remain extended and stable during the hop",
                ],
            },
            "3rd": {
                "talam": "Aadhi Taalam",
                "syllables": "Taiyum Tat tat / Taiyum Tam",
                "right_side": {
                    "key_poses": [
                        "(a) Taiyum: Extend right leg outward, arms extended to sides",
                        "(b) Tat tat: Strike right foot, arms extended horizontally",
                        "(c) Taiyum: Extend left leg outward, arms extended",
                        "(d) Tam: Strike left leg, arms extended",
                    ],
                    "notes": "Movements are similar to 1st Nattadavu (Right and Left side).",
                },
                "left_side": {
                    "key_poses": [
                        "(a) Taiyum: Extend left leg outward, arms extended",
                        "(b) Tat tat: Lift and strike left leg",
                        "(c) Taiyum: Cross left, lift and strike right leg, turn left wrist",
                        "(d) Tam: Lift and extend left leg, lift and strike left leg",
                    ],
                    "notes": "Movements are similar to right side, but start with left leg and left hand.",
                },
                "extended_movements": [
                    "(e) Taiyum: Cross right toes and fold right arm to chest level / Tat tat: Lift and strike left leg, turn right wrist",
                    "(f) Taiyum: Strike right toes, turn right wrist / Tam: Lift and strike left leg, turn right wrist",
                    "(g) Taiyum: Lift and extend right leg, pull right shoulder inward / Tat tat: Lift and strike right leg, push right shoulder out, turn right wrist",
                    "(h) Taiyum: Lift and extend right leg, pull right shoulder inward / Tam: Lift and strike right leg, push right shoulder out, turn down right wrist",
                ],
                "technique_points": [
                    "Arms must be fully extended horizontally at shoulder height",
                    "Wrist turns (rechaka) should be crisp and deliberate",
                    "Leg extensions should reach full stretch before striking",
                    "Shoulder movements coordinate with arm positions",
                    "Maintain aramandi depth even while extending legs laterally",
                    "Torso remains upright during all lateral movements",
                    "Left-right symmetry: left side should mirror right side exactly",
                    "Cross-body movements require coordinated shoulder rotation",
                ],
            },
            "4th": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Yum Tat Ta Tai Hi / Tai Yum Tam",
                "key_poses": [
                    "Aramandi with arms extended",
                    "Extend leg outward with double strike pattern",
                    "Arm crosses body to opposite side during extension",
                    "Return to center with wrist turn",
                ],
                "technique_points": [
                    "Introduces cross-body arm movement with leg extension",
                    "The arm crossing must be at shoulder height, not dropping",
                    "Double strike (Tat Ta) requires quick, precise footwork",
                    "Balance is more challenging with cross-body coordination",
                    "Torso rotation should be minimal - movement is in the limbs",
                ],
            },
            "5th": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Yum Tat Ta Tai Hi / Tai Yum Tam",
                "key_poses": [
                    "Aramandi with one arm extended, other arm bent at chest",
                    "Extend leg outward while alternating arm positions",
                    "Strike with coordinated arm switch",
                    "Wrist turns on each arm transition",
                ],
                "technique_points": [
                    "Asymmetric arm positions require strong body awareness",
                    "The bent arm at chest should have a precise hasta (hand gesture)",
                    "Arm switch should be simultaneous with the foot strike",
                    "Shoulder line must stay level despite asymmetric arms",
                    "Wrist rechaka on each arm change must be clearly visible",
                ],
            },
            "6th": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Yum Tat Ta Tai Hi / Tai Yum Tam",
                "key_poses": [
                    "Aramandi with arms in alternating diagonal positions",
                    "One arm extended up diagonally, other down diagonally",
                    "Leg extension with diagonal arm coordination",
                    "Reverse diagonal on opposite side",
                ],
                "technique_points": [
                    "Diagonal arm lines create the distinctive nattadavu 6 silhouette",
                    "Upper arm reaches above shoulder, lower arm below - both fully extended",
                    "The diagonal must be a clean straight line through both hands",
                    "Leg extension direction aligns with arm positioning",
                    "Transitioning diagonals between sides must be smooth and controlled",
                    "Aramandi depth maintained despite the added upper body complexity",
                ],
            },
            "7th": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Yum Tat Ta Tai Hi / Tai Yum Tam",
                "key_poses": [
                    "Aramandi with arms extended, complex wrist work",
                    "Leg extends with shoulder pull inward",
                    "Strike with shoulder push outward and wrist turn",
                    "Cross-body coordination with turning movements",
                ],
                "technique_points": [
                    "Shoulder push-pull adds rotational element to the movement",
                    "Wrist turns become multi-directional (up, down, outward)",
                    "The shoulder rotation must originate from the upper back, not the waist",
                    "Leg extension timing must sync precisely with shoulder movement",
                    "Advanced coordination - each limb has its own pattern that must merge",
                ],
            },
            "8th": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Yum Tat Ta Tai Hi / Tai Yum Tam",
                "key_poses": [
                    "Full combination of all nattadavu elements",
                    "Lateral extension with cross-body arms and shoulder rotation",
                    "Multiple wrist turns per phrase",
                    "Complex leg patterns with hops and cross-steps",
                ],
                "technique_points": [
                    "The most complex nattadavu - integrates all previous elements",
                    "Demands simultaneous control of arms, wrists, shoulders, legs, and torso",
                    "Spatial awareness critical - movements cover more stage space",
                    "Rhythm must remain precise despite the complexity",
                    "Aramandi is the anchor - if it weakens, the entire adavu suffers",
                    "Symmetry between sides becomes the ultimate test of mastery",
                ],
            },
        },
    },
    "paraval_adavu": {
        "description": (
            "Paraval adavu (also called Parachal adavu) involves sliding or gliding "
            "footwork. The dancer slides one foot outward along the floor while "
            "maintaining aramandi, then brings it back. Arms move in coordination, "
            "typically with one arm extended and the other at the chest."
        ),
        "variations": {
            "1st": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Ya Tai Hi",
                "key_poses": [
                    "Aramandi with right arm extended, left at chest",
                    "Slide right foot outward along floor, maintaining aramandi depth",
                    "Bring right foot back to center",
                    "Mirror with left side",
                ],
                "technique_points": [
                    "The slide must be smooth and continuous, not jerky",
                    "Foot stays in contact with the floor throughout the slide",
                    "Aramandi depth must not change during the slide",
                    "The extended arm follows the direction of the sliding foot",
                    "Weight shifts gradually toward the sliding foot then back to center",
                    "Torso remains upright - no leaning toward the sliding direction",
                ],
            },
        },
    },
    "tattimetti": {
        "description": (
            "Tattimetti adavu combines a heel strike with a toe strike in alternation. "
            "The dancer strikes with the heel (metti) and then the flat foot (tatti), "
            "creating a distinctive two-part rhythmic pattern. Hands typically remain "
            "on the waist or move in simple coordinated patterns."
        ),
        "variations": {
            "1st": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Hat Tai Hi / Tat Tai Tam",
                "key_poses": [
                    "Aramandi with hands on waist",
                    "Heel strike (metti) - heel contacts floor with toes raised",
                    "Flat foot strike (tatti) - full foot contacts floor",
                    "Alternating heel-flat pattern between feet",
                ],
                "technique_points": [
                    "Clear distinction between heel strike and flat strike is essential",
                    "Heel strike: only the heel touches, toes are visibly lifted",
                    "Flat strike: entire foot lands firmly and flat",
                    "The two-part strike creates a characteristic sound pattern",
                    "Aramandi must be maintained between both types of strikes",
                    "Weight must be controlled during heel strikes (less stable than flat)",
                ],
            },
        },
    },
    "kuditta_mettu": {
        "description": (
            "Kuditta Mettu adavu involves jumping or stamping movements. The dancer "
            "jumps and lands in aramandi, often with coordinated arm movements. "
            "This adavu builds strength, control, and the ability to maintain form "
            "during aerial movements."
        ),
        "variations": {
            "1st": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Tai Tam / Tat Tai Tam",
                "key_poses": [
                    "Aramandi starting position",
                    "Small controlled jump from aramandi",
                    "Land back in aramandi with flat foot strike",
                    "Arms may extend outward during jump, return to position on landing",
                ],
                "technique_points": [
                    "Jump height should be controlled and consistent, not maximal",
                    "Takeoff and landing must both be from/to proper aramandi",
                    "Landing should be soft and controlled with bent knees absorbing impact",
                    "Upper body stays upright during the jump - no forward lean",
                    "Arms coordinate with the jump but should not flail",
                    "Both feet should leave and land simultaneously",
                ],
            },
        },
    },
    "mandi_adavu": {
        "description": (
            "Mandi adavu involves deep bending movements, going lower than standard "
            "aramandi into a full or near-full sitting position (mandi). The dancer "
            "descends into a deep squat and rises, often with arm movements that "
            "accompany the vertical motion."
        ),
        "variations": {
            "1st": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Ya Tai Hi",
                "key_poses": [
                    "Start in aramandi",
                    "Descend into deep mandi (full squat) with controlled movement",
                    "Arms may sweep downward or outward during descent",
                    "Rise back to aramandi with equal control",
                ],
                "technique_points": [
                    "The descent must be slow, controlled, and graceful - never a drop",
                    "Knees track outward over toes throughout the descent",
                    "Back remains straight and upright even in the deepest position",
                    "The rise should mirror the descent in speed and control",
                    "Weight stays centered between both feet - no shifting to one side",
                    "This adavu builds the leg strength fundamental to all Bharatanatyam",
                ],
            },
        },
    },
    "sarikkal": {
        "description": (
            "Sarikkal adavu involves traversal or traveling movements across the stage. "
            "The dancer moves laterally or diagonally while maintaining aramandi and "
            "performing coordinated arm and leg movements. It teaches spatial awareness "
            "and maintaining form while in motion."
        ),
        "variations": {
            "1st": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Ya Tai Hi",
                "key_poses": [
                    "Aramandi with arms extended to sides",
                    "Step right foot to the right in aramandi",
                    "Bring left foot to meet right, maintaining aramandi",
                    "Continue traveling in same direction, then reverse",
                ],
                "technique_points": [
                    "Aramandi depth must stay constant while traveling",
                    "The head should stay at the same level - no bobbing up and down",
                    "Steps should be smooth and gliding, not stomping",
                    "Arms maintain their position steady during travel",
                    "Direction changes should be crisp and immediate",
                    "Spatial awareness: the dancer should cover consistent distance per step",
                ],
            },
        },
    },
    "pakka_adavu": {
        "description": (
            "Pakka adavu focuses on side-to-side weight shifting movements. The dancer "
            "shifts weight decisively from one foot to the other, often with the "
            "non-weight-bearing foot lifted. Arms typically move in opposition to "
            "the weight shift direction."
        ),
        "variations": {
            "1st": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Ya Tai Hi",
                "key_poses": [
                    "Weight on right foot in aramandi, left foot slightly lifted",
                    "Shift weight decisively to left foot, right foot lifts",
                    "Arms swing in opposition to the weight shift",
                    "Each shift is a distinct, controlled movement",
                ],
                "technique_points": [
                    "Weight transfer must be complete - fully on one foot at a time",
                    "The lifted foot should be clearly off the ground but controlled",
                    "Aramandi depth maintained on the standing leg",
                    "Hips stay level during weight transfer - no tilting",
                    "Arms provide counterbalance and aesthetic line",
                    "Each shift aligns precisely with the beat",
                ],
            },
        },
    },
    "tirmanam": {
        "description": (
            "Tirmanam (also Teermanam) are concluding sequences that end a dance phrase "
            "or section. They involve accelerating rhythmic patterns that culminate in "
            "a strong final pose. Tirmanams test the dancer's ability to maintain form "
            "and precision at increasing speeds."
        ),
        "variations": {
            "1st": {
                "talam": "Aadhi Taalam",
                "syllables": "Tai Tai Tam, Tai Tai Tam, Tai Tai Tam",
                "key_poses": [
                    "Aramandi with appropriate arm position for the sequence",
                    "Accelerating foot strikes in aramandi",
                    "Final Tam: strong definitive pose with full body engagement",
                    "The three repetitions accelerate: slow, medium, fast",
                ],
                "technique_points": [
                    "The three-fold repetition (first slow, then medium, then fast) is the hallmark",
                    "Form must NOT deteriorate as speed increases",
                    "The final Tam of the last repetition is the most important moment",
                    "Final pose must be held with complete stillness and balance",
                    "Aramandi depth often deepens slightly on the final Tam for emphasis",
                    "All energy channels into the concluding pose - it should feel definitive",
                ],
            },
        },
    },
}


def get_adavu_reference(item_type: str | None, item_name: str | None) -> str | None:
    """Build a reference text block for the LLM based on the performance metadata.

    Searches the catalog for matching adavu types and returns formatted
    reference text with technique points and beat patterns.
    """
    if not item_type and not item_name:
        return None

    search_text = f"{item_type or ''} {item_name or ''}".lower()
    matched_sections = []

    for adavu_key, adavu_data in ADAVU_CATALOG.items():
        if adavu_key in search_text:
            matched_sections.append(_format_adavu(adavu_key, adavu_data, search_text))
            continue

        # Also check if any variation number matches the item name
        for var_key in adavu_data.get("variations", {}):
            variation_name = f"{var_key} {adavu_key}"
            if variation_name in search_text:
                matched_sections.append(
                    _format_variation(adavu_key, var_key, adavu_data)
                )

    if not matched_sections:
        # Return general reference if no specific match
        return _format_general_reference()

    return "\n\n".join(matched_sections)


def _format_adavu(adavu_key: str, adavu_data: dict, search_text: str) -> str:
    """Format a full adavu entry with all its variations."""
    lines = [
        f"## Reference: {adavu_key.title()}",
        adavu_data["description"],
    ]

    variations = adavu_data.get("variations", {})

    # If a specific variation is mentioned, only include that one
    matched_var = None
    for var_key in variations:
        if var_key in search_text:
            matched_var = var_key
            break

    if matched_var:
        lines.append(_format_variation_detail(matched_var, variations[matched_var]))
    else:
        # Include all variations
        for var_key, var_data in variations.items():
            lines.append(_format_variation_detail(var_key, var_data))

    return "\n".join(lines)


def _format_variation(adavu_key: str, var_key: str, adavu_data: dict) -> str:
    """Format a specific variation."""
    lines = [
        f"## Reference: {var_key} {adavu_key.title()}",
        adavu_data["description"],
        _format_variation_detail(var_key, adavu_data["variations"][var_key]),
    ]
    return "\n".join(lines)


def _format_variation_detail(var_key: str, var_data: dict) -> str:
    """Format the detail block for a single variation."""
    lines = [
        f"\n### {var_key} variation",
        f"- Talam: {var_data.get('talam', 'N/A')}",
        f"- Syllables: {var_data.get('syllables', 'N/A')}",
    ]

    if "speeds" in var_data:
        lines.append("- Beat pattern (1st speed): " + var_data["speeds"].get("1st", ""))

    if "key_poses" in var_data:
        lines.append("- Key poses:")
        for pose in var_data["key_poses"]:
            lines.append(f"  - {pose}")

    # Include right/left side details if present (nattadavu)
    for side in ("right_side", "left_side"):
        side_data = var_data.get(side)
        if side_data:
            label = side.replace("_", " ").title()
            lines.append(f"- {label}:")
            for pose in side_data.get("key_poses", []):
                lines.append(f"  - {pose}")

    if "technique_points" in var_data:
        lines.append("- Technique checkpoints:")
        for point in var_data["technique_points"]:
            lines.append(f"  - {point}")

    if "notes" in var_data:
        lines.append(f"- Note: {var_data['notes']}")

    return "\n".join(lines)


def _format_general_reference() -> str:
    """Return general Bharatanatyam technique reference when no specific adavu matches."""
    return """## General Bharatanatyam Technique Reference

Key posture elements to evaluate:
- **Aramandi**: The fundamental half-sitting position. Knees bent outward over toes, back straight, weight centered.
- **Natyarambhe**: Hands on waist position used in tattadavus. Elbows out, shoulders down and back.
- **Arm extension**: In nattadavu and other adavus, arms should be fully stretched at shoulder height.
- **Foot strikes**: Should be flat and firm, creating clear sound. Heel-to-toe coordination matters.
- **Torso**: Must remain upright and stable. No leaning forward/backward during footwork.
- **Symmetry**: All movements performed on one side must be mirrored exactly on the other.
- **Wrist turns (rechaka)**: Crisp, deliberate rotations coordinated with arm/leg movements.
- **Weight distribution**: Should be even in symmetric positions, controlled during transitions."""
