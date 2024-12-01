from nicegui import ui
from src.ui import theme, dashboard_tab, trends_tab, prediction_tab
from fastapi import FastAPI


def init(fastapi_app: FastAPI) -> None:

    @ui.page("/")
    def main_page():
        # Cloudflare Web Analytics snippet
        ui.add_body_html("""
            <!-- Cloudflare Web Analytics -->
            <script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "6b13112733624e9ab542405330f348c8"}'></script>
            <!-- End Cloudflare Web Analytics -->
        """)
        with theme.frame() as tabs:
            with ui.tab_panels(tabs=tabs, value="Dashboard").classes(
                "w-full h-full justify-center items-center"
            ) as tab_panels:
                with ui.tab_panel("Dashboard"):
                    dashboard_instance = dashboard_tab.DashBoardTab()
                    dashboard_instance.display_tab()
                with ui.tab_panel("Trends"):
                    trends_instance = trends_tab.TrendsTab()
                    trends_instance.display_tab()
                with ui.tab_panel("Prediction"):
                    prediction_instance = prediction_tab.PredictionTab()
                    prediction_instance.display_tab()

    ui.run_with(
        fastapi_app,
        title="Belgian Housing Market Insights",
        favicon="src/ui/assets/belgium.png",
    )
