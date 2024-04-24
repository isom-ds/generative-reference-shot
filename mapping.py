# Mapped using https://link.springer.com/article/10.1007/s00357-019-9307-0

map_ekman = [
    'anger',
    'disgust',
    'fear',
    'joy',
    'sadness',
    'surprise'
]

map_affectivetext = {
    # anger | anticipation | disgust | fear | joy | love | optimism | pessimism | sadness | surprise | trust | 
    'anger': {'primary': ['anger'], 'secondary': ['anger']},
    'disgust': {'primary': ['disgust'], 'secondary': ['disgust']},
    'fear': {'primary': ['fear'], 'secondary': ['fear']},
    'joy': {'primary': ['joy'], 'secondary': ['joy', 'love', 'optimism', 'anticipation']},
    'sadness': {'primary': ['sadness'], 'secondary': ['sadness', 'pessimism']},
    'surprise': {'primary': ['surprise'], 'secondary': ['surprise', 'trust']}
}

map_crowdflower = {
    # sadness | enthusiasm | neutral | worry | surprise | love | fun | hate | happiness | boredom | relief | anger
    'anger': {'primary': ['anger'], 'secondary': ['anger', 'hate']},
    'disgust': {'primary': [], 'secondary': []},
    'fear': {'primary': [], 'secondary': ['worry']},
    'joy': {'primary': ['happiness'], 'secondary': ['happiness', 'enthusiasm', 'love', 'fun', 'relief']},
    'sadness': {'primary': ['sadness'], 'secondary': ['sadness', 'boredom']},
    'surprise': {'primary': ['surprise'], 'secondary': ['surprise']}
}

map_electoraltweets = {
    # disgust | anger_or_annoyance_or_hostility_or_fury | anticipation_or_expectancy_or_interest | admiration | dislike
    # uncertainty_or_indecision_or_confusion | joy_or_happiness_or_elation | like | indifference | disappointment 
    # calmness_or_serenity | hate | amazement | acceptance | fear_or_apprehension_or_panic_or_terror | vigilance | trust 
    # surprise | sadness_or_gloominess_or_grief_or_sorrow
    'anger': {'primary': ['anger_or_annoyance_or_hostility_or_fury'], 'secondary': ['anger_or_annoyance_or_hostility_or_fury', 'hate']},
    'disgust': {'primary': ['disgust'], 'secondary': ['disgust', 'dislike']},
    'fear': {'primary': ['fear_or_apprehension_or_panic_or_terror'], 'secondary': ['fear_or_apprehension_or_panic_or_terror', 'uncertainty_or_indecision_or_confusion', 'vigilance']},
    'joy': {'primary': ['joy_or_happiness_or_elation'], 'secondary': ['joy_or_happiness_or_elation', 'anticipation_or_expectancy_or_interest', 'admiration', 'like', 'calmness_or_serenity', 'acceptance']},
    'sadness': {'primary': ['sadness_or_gloominess_or_grief_or_sorrow'], 'secondary': ['sadness_or_gloominess_or_grief_or_sorrow', 'disappointment']},
    'surprise': {'primary': ['surprise'], 'secondary': ['surprise', 'amazement', 'trust']} 
}


map_ssec = {
    # anger | anticipation | disgust | fear | joy | sadness | surprise | trust
    'anger': {'primary': ['anger'], 'secondary': ['anger']},
    'disgust': {'primary': ['disgust'], 'secondary': ['disgust']},
    'fear': {'primary': ['fear'], 'secondary': ['fear']},
    'joy': {'primary': ['joy'], 'secondary': ['joy', 'anticipation']},
    'sadness': {'primary': ['sadness'], 'secondary': ['sadness']},
    'surprise': {'primary': ['surprise'], 'secondary': ['surprise', 'trust']}
}