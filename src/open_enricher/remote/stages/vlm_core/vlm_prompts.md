Analyze the sampled frames of the video and extract evidence-grounded metadata for a video recommender system.

The frames are a sparse sample spread across the whole file, from the opening through the ending. Describe what those frames actually show. Cautious wording is required for guesses, but visible people, places, actions, lighting, and on-screen text must be written down. An all-empty JSON object is an invalid answer.

Rules:
- Do not invent dialogue, official titles, or character names that the frames do not show. A descriptive title_guess is still required.
- Use an empty string or an empty array only for a single field that the frames cannot support. Do not leave title_guess, short_summary, tags, genres, mood, visual_style, opening_scene, or ending_scene empty.
- scene_timeline must contain several segments spaced across the whole file. Each segment needs a non-empty description. The last segment ends near the end of the file.
- setting is one string. notable_scenes is a list of strings. Do not nest arrays inside those fields.
- Lists hold at most 8 short items. After the last item, close the array and write the next key. Do not describe one object piece by piece. Do not repeat an item or extend it with the words "content" or "visible".
- Return one JSON object and no other text. No Markdown fences. Include every key below. Replace every placeholder with real content.

Keys and types:
{
  "title_guess": "short descriptive title",
  "short_summary": "what happens across the samples",
  "tags": ["search tag"],
  "notable_objects": ["visible object"],
  "notable_scenes": ["timestamp and what is visible"],
  "content_type": "for example film, episode, or trailer",
  "primary_category": "broad category",
  "genres": ["genre"],
  "topics": ["topic"],
  "themes": ["theme"],
  "activities": ["visible activity"],
  "setting": "places that are visible",
  "people_or_characters": ["who is visible, without invented names"],
  "narrative_structure": "how the samples progress",
  "opening_scene": "what the early frames show",
  "ending_scene": "what the late frames show",
  "scene_timeline": [
    {
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS",
      "description": "what this stretch shows",
      "subjects_and_objects": ["visible subject"],
      "actions": ["visible action"],
      "camera_and_shot": "framing if discernible",
      "lighting_and_color": "light and palette if discernible"
    }
  ],
  "visual_style": "image style and color",
  "mood": ["mood"],
  "pacing": "Fast, Moderate, or Slow",
  "on_screen_text": ["readable text, or empty if none is readable"],
  "recommendation_signals": {
    "similarity_keywords": ["keyword"],
    "likely_viewing_interests": ["content interest, not a claim about a person"],
    "distinctive_appeal_factors": ["what stands out visually"],
    "content_intensity": "how intense the images are",
    "content_suitability_notes": "what is depicted that affects suitability"
  },
  "content_warnings": ["warning grounded in a visible image, or empty"],
  "evidence_notes": ["which observation is uncertain"]
}
