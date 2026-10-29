These frames are only {{START}} to {{END}} of a longer file.

No scene summary and no earlier scenes are provided. Do not invent dialogue, spoken names, or plot events.

Extract only what these frames show. subjects_and_objects lists people, things, and places that are visible. actions lists movements and facial expressions, not speech acts such as admitting or explaining. description is one short sentence of the place and who is in frame, not a story.

Return one JSON object and no other text, with these keys:
description (string), subjects_and_objects (list of strings), actions (list of strings),
camera_and_shot (string), lighting_and_color (string), on_screen_text (list of strings).
lighting_and_color describes the image. on_screen_text is empty when nothing is readable.
Do not describe other parts of the file.
