import unittest

from train_panel import HTML_PAGE


class WorkspaceLayoutTests(unittest.TestCase):
    def test_desktop_uses_the_full_viewport(self) -> None:
        self.assertIn(".wrap{width:100%;height:100vh", HTML_PAGE)
        self.assertIn("grid-template-columns:214px minmax(0,1fr)", HTML_PAGE)
        self.assertIn("height:calc(100vh - 64px)", HTML_PAGE)

    def test_sidebar_can_collapse_and_remembers_the_choice(self) -> None:
        self.assertIn('id="sidebar-toggle"', HTML_PAGE)
        self.assertIn("body.sidebar-collapsed .layout", HTML_PAGE)
        self.assertIn("function toggleSidebar()", HTML_PAGE)
        self.assertIn("yoloTeamPlatformSidebarCollapsed", HTML_PAGE)

    def test_mobile_falls_back_to_document_scrolling(self) -> None:
        self.assertIn("html,body{height:auto;overflow:auto}", HTML_PAGE)
        self.assertIn(".layout,body.sidebar-collapsed .layout{height:auto;grid-template-columns:1fr", HTML_PAGE)


if __name__ == "__main__":
    unittest.main()
