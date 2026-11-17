---
when: making a poster, cover, panel, logo or any image for Contravía, the game's commercial name; changing the brand's colours, fonts or logo; generating art with FLUX.2 klein for the game
---

# Branding: Contravía

Contravía is the game's name, and `contravia` (no accent) names the repo, the package, the CLI, release assets and the data folder; the project was called Caimán until 2026-11-17. A *contravía* is driving the wrong
way up a one-way street: a traffic offence that names an attitude toward the car and no story, so it fits any
driving gameplay and any city. The game covers all of Cuba from the start, so no piece names one city
in the logo; a city or a route goes in the tagline (today "DE PINAR DEL RÍO A BARACOA").

The brand is for private showings for now. Everything lives in `branding/`.

## What the brand is made of

| Part | Where | Value |
|---|---|---|
| Yellow (the word) | `branding/brand.css` `--cv-yellow` | `#ffd400` |
| Red (sign, drop shadow) | `--cv-red` | `#d62828` |
| Ink (background, outlines, gutters) | `--cv-ink` | `#0b0b0b` |
| Paper (frames, sign bar, tagline) | `--cv-paper` | `#f4efe4` |
| Logo font | `branding/fonts/NotoSans-CondensedBlackItalic.ttf` | the word, skewed 6° more, yellow, ink stroke, red shadow |
| Tagline font | `branding/fonts/NotoSans-CondensedBlack.ttf` | uppercase, wide letter spacing |
| Text font | `branding/fonts/NotoSans-CondensedBold.ttf` | captions and balloons |

The fonts are Noto Sans (SIL Open Font License), so they ship with the repo. The logo is HTML and CSS
(`.cv-logo`, `.cv-sign`, `.cv-word`, `.cv-tag` in `brand.css`), never a generated image: an image model draws
letters badly, and the CSS logo scales with `--cv-size`.

## The look

- Panels are bold hand-painted cover illustrations: thick black ink outlines, cel shading, saturated colours,
  dramatic low angles, cinematic light. No text, letters, numbers or logos inside a panel; all text is set in HTML.
- A cover is a collage of panels of different sizes on an ink background with paper frames, and the logo under it.
- Show the whole island: Capitolio, Malecón, Trinidad, Santiago, a stop at the Plaza de la Revolución.
- Show the vehicles of Cuba today: almendrones (1950s American sedans), Ladas, the yellow electric passenger
  tricycles, the articulated metroguaguas, electric scooters, motorcycles with sidecars. Camellos left Havana
  in 2008: do not draw them.
- Inspired by open-world game covers, never copied from one: no font, logo, name pattern or exact panel layout of
  an existing game, no real car maker's badge or name. Prompts ask for cars "with no badges".

## Making a panel

Panels are made with FLUX.2 klein 4B on our GPU machine (the workspace know-how on GPU machines says how to reach
it), never with a paid model. Its batch script loads the model once and takes a JSON list of jobs:
`{"out", "prompt", "refs", "w", "h", "seed", "steps"}`; a job whose output exists is skipped.

1. Write the prompt as the scene, then the style sentence every panel shares (copy it from `branding/panels.json`).
2. Pick the size from the slot it fills: 1344x768 for wide, 768x1152 for tall, 1024x1024 for square, 1344x640 for
   a band. Each image takes about 8 s, or 10 to 20 s with references.
3. When klein does not know the thing (a Cuban vehicle, a landmark), give it one or two reference photos. It keeps
   their shapes and colours and repaints them in the panel style. The photos are only inputs; none appears in
   a panel. Write each one's URL in the job; licences matter only if the image is ever published.
4. Run two seeds of each panel and keep the better one. Write the kept seed in `branding/panels.json`, which holds
   the prompt, size, seed and references of every panel on the poster, so any panel can be made again.
5. Save the kept panel as JPEG (quality 90) in `branding/poster/panels/`.

A game screenshot also works as a reference: klein repaints the city the game generates in the panel style.
Crop the HUD and the minimap off first.

## Making the poster

`branding/poster/poster.html` places the panels with a CSS grid (`grid-template-areas`; tracks must be
`minmax(0, 1fr)`, or the images grow them) and puts the logo under it. Change a panel, the order or the tagline
there; no image needs to be made again for that. Render it with
`uv run --with playwright python branding/render.py branding/poster/poster.html branding/poster/contravia-poster.jpg`,
which fails if an image did not load.
