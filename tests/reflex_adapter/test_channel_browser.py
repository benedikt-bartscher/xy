"""A chart rendering in a real browser over the Reflex channel transport.

The rest of the adapter suite drives the data plane from Python. This one
proves the thing that only a browser can: the wrapper opens no connection of
its own, the app's single websocket carries the chart's binary columns, and
the WebGL view paints from them.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest

pytest.importorskip("playwright")

from PIL import Image  # noqa: E402
from playwright.sync_api import WebSocket, sync_playwright  # noqa: E402
from reflex.testing import AppHarness  # noqa: E402

# The app's event websocket, the one the chart has to share.
EVENT_PATH = "/_event"

# Headless chromium has no GPU; xy's own browser probes use the same pair.
CHROMIUM_ARGS = ("--use-angle=swiftshader", "--enable-unsafe-swiftshader")


def ChannelChartApp():
    """A Reflex app with one live xy chart served over a channel."""
    import numpy as np
    import reflex as rx

    import reflex_xy
    import xy

    def orbits() -> "xy.Chart":
        rng = np.random.default_rng(3)
        count = 20_000
        theta = rng.uniform(0.0, 2.0 * np.pi, count)
        radius = rng.normal(1.0, 0.05, count)
        return xy.scatter_chart(
            xy.scatter(radius * np.cos(theta), radius * np.sin(theta), opacity=0.6),
            xy.x_axis(label="x"),
            xy.y_axis(label="y"),
            width="100%",
            height=240,
        )

    token = reflex_xy.inline(orbits())

    @rx.page("/")
    def index():
        return rx.box(
            reflex_xy.chart(token, height="240px"),
            rx.text("ready", id="ready"),
        )

    app = rx.App()
    reflex_xy.setup(app)


@pytest.fixture
def _fresh_registry():
    """Keep the registry across this module's tests.

    The package-wide autouse fixture resets it between tests, but this app
    registers its figure once, when the harness imports it — a reset would
    leave the subscribe with no figure to serve.

    Yields:
        None; this only shadows the resetting fixture.
    """
    yield


@pytest.fixture(scope="module")
def chart_app(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[AppHarness, None, None]:
    """Build and serve the chart app.

    Args:
        tmp_path_factory: pytest fixture for creating temporary directories.

    Yields:
        The running harness.
    """
    with AppHarness.create(
        root=tmp_path_factory.mktemp("channel_chart_app"),
        app_source=ChannelChartApp,
    ) as harness:
        assert harness.app_instance is not None, "app is not running"
        yield harness


def test_chart_paints_from_channel_binary(chart_app: AppHarness, tmp_path: Path):
    """The chart paints, and its columns arrive on the app's own websocket.

    Args:
        chart_app: The running harness.
        tmp_path: Where to drop a screenshot of the painted chart.
    """
    assert chart_app.frontend_url is not None
    sockets: list[WebSocket] = []
    binary_frames: list[int] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=list(CHROMIUM_ARGS))
        page = browser.new_page(viewport={"width": 1024, "height": 768})

        def watch(socket: WebSocket) -> None:
            sockets.append(socket)
            if EVENT_PATH not in socket.url:
                return  # the dev server's hot-reload channel
            socket.on(
                "framereceived",
                lambda payload: (
                    binary_frames.append(len(payload)) if isinstance(payload, bytes) else None
                ),
            )

        page.on("websocket", watch)
        page.goto(chart_app.frontend_url)
        page.wait_for_selector("#ready")
        # The chart only reaches "ready" once a payload has been applied.
        # A cold dev server still has xy's WebGL client bundle to transform,
        # which outlasts the default wait by a wide margin.
        chart = page.locator('[data-xy-slot="root"]')
        chart.wait_for(state="visible", timeout=120_000)
        page.wait_for_function(
            "!!document.querySelector('[data-xy-context-state=\"ready\"]')",
            timeout=120_000,
        )
        # The accessible summary is computed from the decoded columns.
        summary = page.locator('[id$="-summary"]').inner_text()
        shot = tmp_path / "chart.png"
        chart.screenshot(path=str(shot))
        browser.close()

    # One connection to the backend for the whole page: the chart multiplexes
    # onto the app's event websocket instead of opening a data plane of its
    # own. (The other socket is the dev server's hot-reload channel, which a
    # built app does not have.)
    backend_sockets = [socket.url for socket in sockets if EVENT_PATH in socket.url]
    assert len(backend_sockets) == 1, backend_sockets
    # Columns travelled as binary frames on that socket, not as JSON numbers
    # or base64.
    assert binary_frames, "no binary frame arrived on the app websocket"
    assert max(binary_frames) > 10_000, (
        f"binary frames look too small for 20k points: {sorted(binary_frames)[-3:]}"
    )
    # Axis ranges in the accessible summary are derived from the columns, so
    # they only exist if the binary payload decoded into real data.
    assert "ranges from" in summary, summary
    # And the chart actually painted: a blank mount is one flat colour.
    with Image.open(shot) as image:
        colours = len(image.convert("RGB").getcolors(maxcolors=1 << 20) or [])
    assert colours > 50, f"chart looks blank: {colours} distinct colours"
    print(
        json.dumps(
            {
                "summary": summary.replace("\n", " ")[:120],
                "colours": colours,
                "binary_frames": binary_frames,
            }
        )
    )
