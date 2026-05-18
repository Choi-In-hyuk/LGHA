"""
task name (LIBERO HDF5 파일명) → action_type ID 매핑.
키워드 기반 rule-matching으로 별도 모델 없이 동작한다.
"""

ACTION_TYPES = {
    0: "pick_and_place",
    1: "open_drawer",
    2: "close_drawer",
    3: "open_door",
    4: "close_door",
    5: "push",
    6: "stack",
    7: "turn_on",
    8: "turn_off",
}

NUM_ACTION_TYPES = len(ACTION_TYPES)
ACTION_NAME_TO_ID = {v: k for k, v in ACTION_TYPES.items()}

# keyword rules: (required_keywords, excluded_keywords) → action_type
# 순서가 중요 — 먼저 매칭된 규칙이 적용됨
_RULES = [
    (["open",  "drawer"],          [],            "open_drawer"),
    (["close", "drawer"],          [],            "close_drawer"),
    (["open",  "door"],            [],            "open_door"),
    (["close", "door"],            [],            "close_door"),
    (["open",  "cabinet"],         [],            "open_door"),
    (["close", "cabinet"],         [],            "close_door"),
    (["push"],                     ["drawer"],    "push"),
    (["stack"],                    [],            "stack"),
    (["turn", "on"],               [],            "turn_on"),
    (["turn", "off"],              [],            "turn_off"),
    (["pick",  "place"],           [],            "pick_and_place"),
    (["pick",  "put"],             [],            "pick_and_place"),
    (["pick"],                     [],            "pick_and_place"),
    (["put"],                      [],            "pick_and_place"),
    (["move"],                     [],            "pick_and_place"),
]


def task_name_to_action_type(task_name: str) -> int:
    """
    task_name: HDF5 파일명 또는 LIBERO task 이름
    예) "pick_up_the_butter_and_place_it_in_the_basket_demo"
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet"
    Returns: action_type ID (int)
    """
    name = task_name.lower().replace("_demo", "").replace("-", "_")
    tokens = set(name.split("_"))

    for required, excluded, action_name in _RULES:
        if all(k in tokens for k in required) and not any(k in tokens for k in excluded):
            return ACTION_NAME_TO_ID[action_name]

    # fallback: pick_and_place
    return ACTION_NAME_TO_ID["pick_and_place"]


def extract_target_object(task_name: str) -> str:
    """
    task_name에서 PaliGemma에 넘길 target object 이름을 추출한다.
    예) "pick_up_the_butter_and_place_it_in_the_basket" → "butter"
        "open_the_bottom_drawer_of_the_cabinet"        → "drawer"
    """
    name = task_name.lower().replace("_demo", "").replace("-", "_")

    # KITCHEN_SCENE1_... 형식이면 scene prefix 제거
    parts = name.split("_")
    for i, p in enumerate(parts):
        if p.startswith("scene") and i > 0:
            name = "_".join(parts[i + 1:])
            parts = name.split("_")
            break

    # action verb 이후 "the" 다음 단어가 target
    verbs = {"pick", "put", "open", "close", "push", "stack", "move", "turn", "place"}
    for i, token in enumerate(parts):
        if token in verbs:
            # find "the" after the verb
            for j in range(i + 1, min(i + 5, len(parts))):
                if parts[j] == "the" and j + 1 < len(parts):
                    return parts[j + 1]
            break

    # fallback: 첫 번째 명사처럼 보이는 단어
    stopwords = {"pick", "put", "up", "the", "a", "an", "and", "to", "in",
                 "on", "of", "it", "open", "close", "push", "move", "turn"}
    for token in parts:
        if token not in stopwords and len(token) > 2:
            return token

    return "object"
