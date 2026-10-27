SKILLS = [
    {
        "strategy_id": "alien_biomech",
        "style_family": "sci_fi",
        "title": "外星生物机械",
        "description": "有机与机械融合的外星世界，拥有生物科技表面、活体结构，以及令人不安的人造生长物。",
        "environment": "将自然环境替换为外星生物机械腔体，充满活体墙壁、脉动结构以及人造有机深度。",
        "prop": "将主要可见道具改造为生物机械生长装置，带有肋状外壳、发光脉纹以及有机科技混合逻辑。",
        "lighting": "使用内部生物荧光、脉动器官光线以及低对比外星迷雾，取代日光。",
        "costume": "将服装重新设计为微妙的外星生物机械分层、适应性薄膜以及非人类织物逻辑，同时保持身份可辨识。",
        "subject_surface": "保持面部可识别，同时添加克制的生物科技纹理、半透明光泽以及非人类但身份安全的表面变化。",
        "subject_integration": "将主体置于活体外星系统内，使其光线和接触关系融入生物机械世界。",
        "atmosphere": "使用孢子、漂浮的有机尘埃、潮湿的生物荧光迷雾以及低频环境飘移。",
        "material": "偏好几丁质、半透明薄膜、活体树脂、生物金属以及反应性外星表面。",
        "tags": ["激进", "外星", "有机", "身体科技"],
        "intensity": "激进",
        "pros": ["非常独特的世界逻辑", "道具与世界的高度融合"],
        "cons": ["过度改造主体的风险较高"],
        "forbidden_mixes": ["洁净实验室", "历史奇幻"],
        "anchor_requirements": ["environment_rewrite_headroom", "organic_surface_headroom"],
        "best_for_archetypes": ["world_anchor_driven_scene", "environment_dominant_scene"],
        "high_risk_archetypes": ["identity_safe_face_closeup", "foreground_device_scene"],
        "world_rewrite_cost": "high",
        "identity_risk": "high",
        "prompt_compilation_notes": [
            "Favor environment-wide biological structure changes over small prop-only upgrades.",
            "If identity must stay safe, keep face geometry stable and move the biomech logic into surfaces, costume, and room walls.",
            "Avoid using this style as a light polish; it needs explicit world rewrite and organic material language."
        ],
        "common_failure_modes": ["insufficient_world_rewrite", "identity_overcorruption", "random_biomech_texture_without_structure"],
        "recommended_carriers": ["walls", "background_depth", "costume_layers", "surface_overgrowth"],
    }
]
