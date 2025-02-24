Revisiting the simple map guessing mini game I [made in 2023](https://github.com/m3at/tiny_projects/tree/main/20231107-map_guessing) with LLMs, as a fun way to check models progress.

The page was made with Gemini 2 flash to extract the data from [this wikimedia map](https://commons.wikimedia.org/wiki/File:Blank_map_of_Japan_new.svg) into json (fast and very flexible with its inputs), then Claude for coding the page itself (still the most coding capable model, very subjectively).

The mini-game was usable on first shot –which might not be surprising anymore but is still cool!– then it took about an hour of iteration to polish it. Surprisingly css animation was something Claude struggled with, I ended up writing it by hand.

`index_tailwindcdn.html` is the more readable file with placeholder data, `index.html` embed the css and has the full prefectural data, and `index.min.html` is the minified version.

[Try it here](https://html-preview.github.io/?url=https://github.com/m3at/tiny_projects/blob/main/20250223-japan_province_guessing/index.min.html)

