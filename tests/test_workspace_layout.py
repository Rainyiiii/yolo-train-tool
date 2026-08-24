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

    def test_collapsed_sidebar_uses_semantic_icons_instead_of_numbers(self) -> None:
        icon_names = (
            "overview",
            "projects",
            "train",
            "dataset",
            "models",
            "assets",
            "test",
            "collab",
            "label",
            "convert",
            "logs",
        )
        self.assertEqual(HTML_PAGE.count('class="nav-icon"'), len(icon_names))
        for name in icon_names:
            self.assertIn(f'id="nav-icon-{name}"', HTML_PAGE)
            self.assertIn(f'href="#nav-icon-{name}"', HTML_PAGE)
        self.assertNotIn("项目中心 <span>01</span>", HTML_PAGE)
        self.assertIn('title="协作标注"', HTML_PAGE)
        self.assertIn("body.sidebar-collapsed .nav button .nav-label{display:none}", HTML_PAGE)

    def test_overview_guides_the_complete_engineering_workflow(self) -> None:
        self.assertIn('id="tab-overview"', HTML_PAGE)
        self.assertIn('id="dashboard-next-title"', HTML_PAGE)
        self.assertIn('id="dashboard-pipeline"', HTML_PAGE)
        self.assertIn("function renderDashboard()", HTML_PAGE)
        for page in ("projects", "dataset", "train", "test", "convert"):
            self.assertIn(f'data-page="{page}"', HTML_PAGE)

    def test_low_frequency_features_use_progressive_disclosure(self) -> None:
        self.assertIn('class="experimental-nav"', HTML_PAGE)
        self.assertIn("body.show-experimental .nav button.experimental-nav", HTML_PAGE)
        self.assertIn('class="tool-details"', HTML_PAGE)
        self.assertIn("function toggleExperimentalFeatures", HTML_PAGE)
        self.assertIn("function updateTestSourceUI", HTML_PAGE)
        self.assertIn("function updateExportOptionsUI", HTML_PAGE)

    def test_core_training_controls_remain_visible(self) -> None:
        self.assertIn('<div class="field"><label>基础模型</label>', HTML_PAGE)
        self.assertIn('<div class="field sm"><label>训练轮数</label>', HTML_PAGE)
        self.assertIn('<div class="field sm"><label>批量大小</label>', HTML_PAGE)
        self.assertNotIn('<div class="field advanced-setting"><label>训练设备</label>', HTML_PAGE)
        self.assertIn('class="actions train-primary-bar"', HTML_PAGE)

    def test_project_center_supports_the_full_project_lifecycle(self) -> None:
        self.assertIn('id="project-search"', HTML_PAGE)
        self.assertIn('id="project-health-filter"', HTML_PAGE)
        self.assertIn('class="btn compact edit-project"', HTML_PAGE)
        self.assertIn('class="btn compact duplicate-project"', HTML_PAGE)
        self.assertIn('class="btn compact red delete-project"', HTML_PAGE)
        self.assertIn('id="project-delete-modal"', HTML_PAGE)
        self.assertIn('id="delete-managed-data"', HTML_PAGE)
        self.assertIn("function updateDeleteProjectGuard()", HTML_PAGE)
        self.assertIn("/api/projects/delete", HTML_PAGE)

    def test_mobile_falls_back_to_document_scrolling(self) -> None:
        self.assertIn("html,body{height:auto;overflow:auto}", HTML_PAGE)
        self.assertIn(".layout,body.sidebar-collapsed .layout{height:auto;grid-template-columns:1fr", HTML_PAGE)


if __name__ == "__main__":
    unittest.main()
