import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright
from tqdm import tqdm


async def generate_images(
    n: int, output_dir: Path, base_url: str = "http://localhost:8080"
):
    """Generate n randomized clock images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        print("Preparing browser")
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 600, "height": 600})

        await page.goto(base_url)

        # Wait for assets to load (fonts, textures, HDRIs)
        await page.wait_for_function("window.clockAPI !== undefined", timeout=30000)
        await asyncio.sleep(3)  # Extra time for HDRI loading

        # Hide UI controls for clean screenshots
        await page.evaluate(
            "document.getElementById('controls').style.display = 'none'"
        )

        for i in tqdm(range(n)):
            # Randomize everything
            await page.evaluate("window.clockAPI.randomizeAll()")
            await asyncio.sleep(0.1)  # Brief pause for render

            # Get current time
            time_data = await page.evaluate("window.clockAPI.getCurrentTime()")
            h, m, s = time_data["hours"], time_data["minutes"], time_data["seconds"]

            # Generate filename: index-HH_MM_SS.png
            filename = f"{i:06d}-{h:02d}_{m:02d}_{s:02d}.png"
            filepath = output_dir / filename

            # Take screenshot (canvas only, no UI buttons)
            canvas = page.locator("canvas")
            await canvas.screenshot(path=str(filepath))

        await browser.close()

    print(f"Generated {n} images in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate clock dataset images")
    parser.add_argument("-n", type=int, default=16, help="Number of images to generate")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("dataset"), help="Output directory"
    )
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    args = parser.parse_args()

    asyncio.run(generate_images(args.n, args.output, args.url))


if __name__ == "__main__":
    main()
