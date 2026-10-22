Analyze the following video metadata.
plot_summary is the high-level plot from the full transcript.
scene_metadata lists each interval. dialogue_description is what was said there. The other fields are what the frames showed.

{{SCRIPT_TEXT}}

Hard limits, which override any longer list in the metadata:
- recsys_metadata.primary_genre is exactly one of: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Education, Fantasy, Horror, Music, Mystery, News, Romance, Sci-Fi, Science Fiction, Sports, Thriller, Vlog, Western.
- recsys_metadata.pacing is exactly one of: Fast, Moderate, Slow.
- recsys_metadata.sub_genres has 1 to 4 items. recsys_metadata.mood has 1 to 4 items.
- search_and_discovery.semantic_tags has 3 to 6 items. search_and_discovery.main_topics has 1 to 4 items.
- audience_profiling.viewer_interests has 1 to 4 items.
- audience_profiling.target_age_group is exactly one of: All Ages, Children, Teens, Young Adults, Gen Z, Millennials, Adults, Family. Do not join two labels with 'and'.
- content_summary.visual_style is one sentence and must mention the dominant palette when the metadata describes dark, near-black, or night colors.
- content_summary.narrative_hook is one sentence. Do not say a missing person was found alive unless the metadata says that without contradiction.
- Use spoken names only when the metadata already contains them.

Populate this exact JSON structure:
{{EXAMPLE_STRUCTURE}}
