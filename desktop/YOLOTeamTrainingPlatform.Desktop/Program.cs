using System.Diagnostics;
using System.Net.Http;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace YOLOTeamTrainingPlatform.Desktop;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        Application.Run(new MainWindow());
    }
}

internal sealed class MainWindow : Form
{
    private const string ProductTitle = "YOLO团队训练平台";
    private const string PanelUrl = "http://127.0.0.1:8989/";
    private readonly WebView2 webView = new() { Dock = DockStyle.Fill };
    private readonly Label status = new() { AutoSize = true, Text = "正在准备本地服务…", ForeColor = Color.FromArgb(106, 78, 78) };
    private readonly Button retryButton = new() { Text = "重新连接", AutoSize = true, Visible = false };
    private readonly HttpClient http = new() { Timeout = TimeSpan.FromSeconds(2) };
    private readonly string installRoot;
    private readonly string appRoot;
    private readonly string workspaceRoot;
    private readonly string pythonPath;
    private bool closing;

    public MainWindow()
    {
        installRoot = ResolveInstallRoot();
        appRoot = Directory.Exists(Path.Combine(installRoot, "App")) ? Path.Combine(installRoot, "App") : installRoot;
        workspaceRoot = Path.Combine(installRoot, "Workspace");
        pythonPath = ResolvePythonPath();

        Text = ProductTitle;
        MinimumSize = new Size(1040, 700);
        Size = new Size(1440, 900);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.White;

        var top = new Panel { Dock = DockStyle.Top, Height = 48, Padding = new Padding(14, 9, 14, 8), BackColor = Color.FromArgb(255, 247, 247) };
        var title = new Label { AutoSize = true, Text = ProductTitle, Font = new Font("Microsoft YaHei UI", 11, FontStyle.Bold), ForeColor = Color.FromArgb(165, 37, 37), Location = new Point(14, 13) };
        status.Location = new Point(220, 15);
        retryButton.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        retryButton.Location = new Point(Width - 130, 9);
        retryButton.Click += async (_, _) => await InitializeAsync();
        top.Controls.Add(title);
        top.Controls.Add(status);
        top.Controls.Add(retryButton);
        top.Resize += (_, _) => retryButton.Left = top.ClientSize.Width - retryButton.Width - 14;

        Controls.Add(webView);
        Controls.Add(top);
        Shown += async (_, _) => await InitializeAsync();
        FormClosing += OnFormClosing;
    }

    private string ResolveInstallRoot()
    {
        var configured = Environment.GetEnvironmentVariable("YOLO_TEAM_PLATFORM_HOME");
        if (!string.IsNullOrWhiteSpace(configured)) return Path.GetFullPath(configured);
        var baseDirectory = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        var parent = Directory.GetParent(baseDirectory)?.FullName;
        if (Path.GetFileName(baseDirectory).Equals("Desktop", StringComparison.OrdinalIgnoreCase) && parent is not null) return parent;
        var cursor = new DirectoryInfo(baseDirectory);
        while (cursor is not null)
        {
            if (File.Exists(Path.Combine(cursor.FullName, "panel_service.py"))) return cursor.FullName;
            cursor = cursor.Parent;
        }
        return baseDirectory;
    }

    private string ResolvePythonPath()
    {
        var candidates = new[]
        {
            Path.Combine(installRoot, "Runtime", "Python", "Scripts", "python.exe"),
            Path.Combine(appRoot, ".venv", "Scripts", "python.exe"),
        };
        return candidates.FirstOrDefault(File.Exists) ?? candidates[0];
    }

    private ProcessStartInfo ServiceCommand(string script, params string[] arguments)
    {
        var info = new ProcessStartInfo
        {
            FileName = pythonPath,
            WorkingDirectory = appRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        info.ArgumentList.Add(Path.Combine(appRoot, script));
        foreach (var argument in arguments) info.ArgumentList.Add(argument);
        info.Environment["YOLO_TEAM_PLATFORM_HOME"] = installRoot;
        info.Environment["YOLO_TEAM_PLATFORM_DATA"] = workspaceRoot;
        info.Environment["PYTHONUTF8"] = "1";
        info.Environment["PYTHONIOENCODING"] = "utf-8";
        return info;
    }

    private async Task InitializeAsync()
    {
        retryButton.Visible = false;
        status.Text = "正在启动本地服务…";
        if (!File.Exists(pythonPath) || !File.Exists(Path.Combine(appRoot, "panel_service.py")))
        {
            ShowFailure("运行组件不完整，请重新运行安装程序。", $"Python: {pythonPath}\nApp: {appRoot}");
            return;
        }

        Directory.CreateDirectory(workspaceRoot);
        try
        {
            using var process = Process.Start(ServiceCommand("panel_service.py", "start", "--no-browser"));
            if (process is null) throw new InvalidOperationException("无法启动服务管理器。");
            await process.WaitForExitAsync();
            if (process.ExitCode != 0)
            {
                var detail = await process.StandardError.ReadToEndAsync();
                throw new InvalidOperationException(string.IsNullOrWhiteSpace(detail) ? "本地服务启动失败。" : detail.Trim());
            }
            await WaitForPanelAsync();
            await InitializeWebViewAsync();
            status.Text = "本地服务已连接";
        }
        catch (Exception exception)
        {
            ShowFailure("平台启动失败。", exception.Message);
        }
    }

    private async Task WaitForPanelAsync()
    {
        var deadline = DateTime.UtcNow.AddSeconds(25);
        Exception? lastError = null;
        while (DateTime.UtcNow < deadline)
        {
            try
            {
                using var response = await http.GetAsync(PanelUrl);
                if (response.IsSuccessStatusCode) return;
            }
            catch (Exception error) when (error is HttpRequestException or TaskCanceledException)
            {
                lastError = error;
            }
            await Task.Delay(400);
        }
        throw new TimeoutException($"等待本地服务超时。{lastError?.Message}");
    }

    private async Task InitializeWebViewAsync()
    {
        var userData = Path.Combine(workspaceRoot, "cache", "webview2");
        Directory.CreateDirectory(userData);
        var environment = await CoreWebView2Environment.CreateAsync(null, userData);
        await webView.EnsureCoreWebView2Async(environment);
        webView.CoreWebView2.Settings.AreDevToolsEnabled = false;
        webView.CoreWebView2.Settings.AreDefaultContextMenusEnabled = true;
        webView.CoreWebView2.Settings.IsStatusBarEnabled = false;
        webView.CoreWebView2.NavigationStarting -= NavigationStarting;
        webView.CoreWebView2.NavigationStarting += NavigationStarting;
        webView.CoreWebView2.DownloadStarting -= DownloadStarting;
        webView.CoreWebView2.DownloadStarting += DownloadStarting;
        webView.Source = new Uri(PanelUrl);
    }

    private void NavigationStarting(object? sender, CoreWebView2NavigationStartingEventArgs args)
    {
        if (!Uri.TryCreate(args.Uri, UriKind.Absolute, out var uri)) return;
        if (uri.Host is "127.0.0.1" or "localhost") return;
        args.Cancel = true;
        Process.Start(new ProcessStartInfo(uri.AbsoluteUri) { UseShellExecute = true });
    }

    private void DownloadStarting(object? sender, CoreWebView2DownloadStartingEventArgs args)
    {
        var downloads = Path.Combine(workspaceRoot, "exports", "downloads");
        Directory.CreateDirectory(downloads);
        var fileName = Path.GetFileName(args.ResultFilePath);
        args.ResultFilePath = UniqueDownloadPath(downloads, fileName);
    }

    private static string UniqueDownloadPath(string directory, string fileName)
    {
        var safeName = string.IsNullOrWhiteSpace(fileName) ? "download.bin" : fileName;
        var candidate = Path.Combine(directory, safeName);
        var stem = Path.GetFileNameWithoutExtension(safeName);
        var extension = Path.GetExtension(safeName);
        var index = 2;
        while (File.Exists(candidate)) candidate = Path.Combine(directory, $"{stem}__{index++:00}{extension}");
        return candidate;
    }

    private void ShowFailure(string message, string detail)
    {
        status.Text = message;
        retryButton.Visible = true;
        MessageBox.Show(this, $"{message}\n\n{detail}\n\n日志目录：{Path.Combine(workspaceRoot, "logs")}", ProductTitle, MessageBoxButtons.OK, MessageBoxIcon.Error);
    }

    private void OnFormClosing(object? sender, FormClosingEventArgs args)
    {
        if (closing) return;
        closing = true;
        try
        {
            using var annotation = Process.Start(ServiceCommand("annotation_service.py", "stop"));
            annotation?.WaitForExit(5000);
            using var panel = Process.Start(ServiceCommand("panel_service.py", "stop"));
            panel?.WaitForExit(5000);
        }
        catch
        {
            // The OS will reclaim the desktop process; service logs retain diagnostics.
        }
    }
}
