The current scene spans {{START}} to {{END}} of a longer file.

Scene summary for this same interval. It is a reading key for the frames, not text to rewrite. It fixes who is involved and which facts this interval is about. It does not describe the image.

{{SCENE_DESCRIPTION}}

Stay with those facts. Use them to recognize the people and the moment, then write what the frames show of that moment and what the spoken words alone cannot give: body actions, objects being handled, facial expressions, posture, and the place around them.

description is a visual account of the current frames. Write at least one sentence when any content is recognizable. If the action or identity is unclear, describe the visible setting, object, or posture. Include visible details absent from the dialogue. Do not retell, quote, or paraphrase the summary. Do not list speech acts such as admitting, explaining, negotiating, or praising. If the summary states a fact the frames do not show, leave it out. If a frame contradicts the summary, trust the frame and say so.

subjects_and_objects lists people and things visible in the frames. Use a name from the summary only for a person the frames show. actions lists movements and expressions you can see, not lines that were only spoken.
{{MEMORY}}
Return one JSON object and no other text, with these keys:
description (string), subjects_and_objects (list of strings), actions (list of strings),
camera_and_shot (string), lighting_and_color (string), on_screen_text (list of strings).
lighting_and_color describes the image. on_screen_text is empty when nothing is readable.
Do not describe other parts of the file. Keep names consistent with the earlier metadata when a person is the same.
