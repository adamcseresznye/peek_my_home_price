from contextlib import contextmanager
from nicegui import ui


@contextmanager
def frame():
    """Custom page frame to share the same styling and behavior across all pages"""
    ui.colors(primary="#1F2937", dark="#192230", dark_page="#192230")
    ui.dark_mode().enable()

    with ui.header().classes(
        "w-full justify-center items-center transition-transform duration-300 fixed"
    ) as header:
        ui.label("Belgian Housing Market Insights").classes(
            "font-bold text-center text-4xl mb-2 sm:text-3xl md:text-4xl lg:text-5xl"
        )
        with ui.tabs().classes("w-full justify-center") as tabs:
            ui.tab("Dashboard", icon="dashboard")
            ui.tab("Trends", icon="trending_up")
            ui.tab("Prediction", icon="signal_cellular_alt")

    with ui.column().classes("w-full items-center pt-2"):
        yield tabs

    # Footer with social media links
    with ui.footer().classes("w-full py-2 bg-gray-800 text-white"):
        with ui.row().classes("w-full items-center justify-between px-4"):
            ui.label("© 2024 Peek My Home Price").classes("text-sm")
            with ui.row().classes("gap-4"):
                with ui.link(
                    "", "https://www.linkedin.com/in/adam-cseresznye/", new_tab=True
                ):
                    ui.add_head_html(
                        '<link href="https://unpkg.com/eva-icons@1.1.3/style/eva-icons.css" rel="stylesheet" />'
                    )

                    ui.icon("eva-linkedin").classes("text-xl")
                with ui.link("", "https://github.com/adamcseresznye", new_tab=True):
                    ui.add_head_html(
                        '<link href="https://unpkg.com/eva-icons@1.1.3/style/eva-icons.css" rel="stylesheet" />'
                    )

                    ui.icon("eva-github").classes("text-xl")
                with ui.link("", "https://x.com/csenye22", new_tab=True):
                    ui.add_head_html(
                        '<link href="https://unpkg.com/eva-icons@1.1.3/style/eva-icons.css" rel="stylesheet" />'
                    )

                    ui.icon("eva-twitter").classes("text-xl")
            ui.label("Made with ❤️ in 🇧🇪").classes("text-sm")

    # Add JavaScript for scroll behavior
    ui.add_head_html(
        """
        <script>
            let lastScrollTop = 0;
            document.addEventListener('scroll', function() {
                const header = document.querySelector('header');
                const currentScroll = window.pageYOffset || document.documentElement.scrollTop;
                
                if (currentScroll > lastScrollTop && currentScroll > 100) {
                    // Scrolling down & past threshold
                    header.style.transform = 'translateY(-100%)';
                } else {
                    // Scrolling up or at top
                    header.style.transform = 'translateY(0)';
                }
                lastScrollTop = currentScroll;
            });
        </script>
    """
    )
