from channels.routing import route

from mainpage import consumers


channel_routing = [
    route('featuresExtraction', consumers.featuresExtraction),
    route('keystroke_update', consumers.keystroke_update),
]