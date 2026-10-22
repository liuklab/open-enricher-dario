These frames are only {{START}} to {{END}} of a longer file.

Describe only what these frames show. Do not invent dialogue, spoken names, or plot events that the frames do not show.
{{MEMORY}}
Return one JSON object and no other text, with these keys:
description (string), subjects_and_objects (list of strings), actions (list of strings),
camera_and_shot (string), lighting_and_color (string), on_screen_text (list of strings).
lighting_and_color describes the image. on_screen_text is empty when nothing is readable.
Do not describe other parts of the file. Keep names consistent with the earlier metadata when a person is the same.
