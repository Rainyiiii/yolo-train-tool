using System.Diagnostics;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace YOLOTeamTrainingPlatform.Desktop;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        using var singleInstance = new Mutex(
            initiallyOwned: true,
            name: @"Local\YOLOTeamTrainingPlatform.Desktop",
            createdNew: out var createdNew);
        if (!createdNew)
        {
            MessageBox.Show(
                "YOLO团队训练平台已经打开，不会重复启动第二个桌面实例。",
                "YOLO团队训练平台",
                MessageBoxButtons.OK,
                MessageBoxIcon.Information);
            return;
        }
        Application.Run(new MainWindow());
        GC.KeepAlive(singleInstance);
    }
}

internal sealed class MainWindow : Form
{
    private sealed record ServiceCommandResult(int ExitCode, string StandardOutput, string StandardError);
    private sealed record RunningPanel(
        [property: JsonPropertyName("pid")] int Pid,
        [property: JsonPropertyName("port")] int Port,
        [property: JsonPropertyName("script")] string Script);

    private const string ProductTitle = "YOLO团队训练平台";
    private const int DefaultPanelPort = 8989;
    private readonly WebView2 webView = new() { Dock = DockStyle.Fill };
    private readonly Label status = new() { AutoSize = true, Text = "正在准备本地服务…", ForeColor = Color.FromArgb(106, 78, 78) };
    private readonly Button retryButton = new() { Text = "重新连接", AutoSize = true, Visible = false };
    private readonly HttpClient http = new() { Timeout = TimeSpan.FromSeconds(2) };
    private readonly string installRoot;
    private readonly string appRoot;
    private readonly string workspaceRoot;
    private readonly string pythonPath;
    private readonly string launcherLogPath;
    private Uri panelUri = new($"http://127.0.0.1:{DefaultPanelPort}/");
    private bool closing;
    private bool ownsPanelService;

    public MainWindow()
    {
        installRoot = ResolveInstallRoot();
        appRoot = Directory.Exists(Path.Combine(installRoot, "App")) ? Path.Combine(installRoot, "App") : installRoot;
        workspaceRoot = Path.Combine(installRoot, "Workspace");
        pythonPath = ResolvePythonPath();
        launcherLogPath = Path.Combine(workspaceRoot, "logs", "launcher.log");

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
            WindowStyle = ProcessWindowStyle.Hidden,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
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
        if (closing) return;
        retryButton.Visible = false;
        status.Text = "正在启动本地服务…";
        if (!File.Exists(pythonPath) || !File.Exists(Path.Combine(appRoot, "panel_service.py")))
        {
            await AppendLauncherLogAsync($"运行组件不完整。Python={pythonPath}; App={appRoot}");
            ShowFailure("运行组件不完整，请重新运行安装程序。", $"Python: {pythonPath}\nApp: {appRoot}");
            return;
        }

        Directory.CreateDirectory(workspaceRoot);
        try
        {
            if (ownsPanelService)
            {
                await RunServiceCommandAsync("panel_service.py", "stop");
                ownsPanelService = false;
            }

            var runningPanels = await ListRunningPanelsAsync();
            if (runningPanels.Count > 0)
            {
                var panelDetails = string.Join(
                    Environment.NewLine,
                    runningPanels.Select(panel => $"PID {panel.Pid} · 端口 {panel.Port}"));
                var choice = MessageBox.Show(
                    this,
                    $"检测到已有 YOLO 训练面板正在运行：\n\n{panelDetails}\n\n为避免同时运行多个服务，是否关闭已有服务并启动当前安装版？",
                    ProductTitle,
                    MessageBoxButtons.YesNo,
                    MessageBoxIcon.Question,
                    MessageBoxDefaultButton.Button2);
                await AppendLauncherLogAsync($"检测到 {runningPanels.Count} 个已有服务；用户选择：{choice}。");
                if (choice != DialogResult.Yes)
                {
                    status.Text = "已保留原有服务，当前窗口即将关闭";
                    Close();
                    return;
                }

                var stopResult = await RunServiceCommandAsync("panel_service.py", "stop-all");
                await LogServiceResultAsync("关闭已有服务", stopResult);
                EnsureServiceCommandSucceeded(stopResult, "无法关闭已有 YOLO 服务。");
                await Task.Delay(500);
                if ((await ListRunningPanelsAsync()).Count > 0)
                {
                    throw new InvalidOperationException("仍有 YOLO 训练面板正在运行，请在任务管理器中关闭后重试。");
                }
            }

            var requestedPort = FindAvailablePanelPort();
            panelUri = new Uri($"http://127.0.0.1:{requestedPort}/");
            await AppendLauncherLogAsync($"准备启动平台。InstallRoot={installRoot}; AppRoot={appRoot}; RequestedPort={requestedPort}");

            var startResult = await RunServiceCommandAsync(
                "panel_service.py", "start", "--no-browser", "--port", requestedPort.ToString());
            await LogServiceResultAsync("启动本地服务", startResult);
            EnsureServiceCommandSucceeded(startResult, "本地服务启动失败。");
            ownsPanelService = true;
            if (closing)
            {
                await RunServiceCommandAsync("panel_service.py", "stop");
                ownsPanelService = false;
                return;
            }
            panelUri = ExtractPanelUri(startResult.StandardOutput) ?? panelUri;
            await WaitForPanelAsync();
            await InitializeWebViewAsync();
            status.Text = $"本地服务已连接 · 端口 {panelUri.Port}";
            await AppendLauncherLogAsync($"平台连接成功。Url={panelUri}");
        }
        catch (Exception exception)
        {
            await AppendLauncherLogAsync($"平台启动失败。{exception}");
            if (!closing) ShowFailure("平台启动失败。", exception.Message);
        }
    }

    private async Task<ServiceCommandResult> RunServiceCommandAsync(string script, params string[] arguments)
    {
        using var process = Process.Start(ServiceCommand(script, arguments));
        if (process is null) throw new InvalidOperationException("无法启动服务管理器。");
        var standardOutputTask = process.StandardOutput.ReadToEndAsync();
        var standardErrorTask = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        return new ServiceCommandResult(
            process.ExitCode,
            (await standardOutputTask).Trim(),
            (await standardErrorTask).Trim());
    }

    private async Task<List<RunningPanel>> ListRunningPanelsAsync()
    {
        var result = await RunServiceCommandAsync("panel_service.py", "list");
        await LogServiceResultAsync("检查已有服务", result);
        EnsureServiceCommandSucceeded(result, "无法检查已有 YOLO 服务，为避免重复启动，当前操作已停止。");
        try
        {
            return JsonSerializer.Deserialize<List<RunningPanel>>(result.StandardOutput) ?? [];
        }
        catch (JsonException exception)
        {
            throw new InvalidOperationException("已有服务检查结果无法解析，为避免重复启动，当前操作已停止。", exception);
        }
    }

    private async Task LogServiceResultAsync(string operation, ServiceCommandResult result)
    {
        await AppendLauncherLogAsync(
            $"{operation}完成。ExitCode={result.ExitCode}\nSTDOUT:\n{result.StandardOutput}\nSTDERR:\n{result.StandardError}");
    }

    private static void EnsureServiceCommandSucceeded(ServiceCommandResult result, string fallbackMessage)
    {
        if (result.ExitCode == 0) return;
        var detail = string.Join(
            Environment.NewLine,
            new[] { result.StandardError, result.StandardOutput }.Where(value => !string.IsNullOrWhiteSpace(value)));
        throw new InvalidOperationException(string.IsNullOrWhiteSpace(detail) ? fallbackMessage : detail);
    }

    private static int FindAvailablePanelPort()
    {
        var candidates = new List<int> { DefaultPanelPort };
        candidates.AddRange(Enumerable.Range(8991, 9));
        candidates.AddRange(Enumerable.Range(9010, 11));
        foreach (var port in candidates)
        {
            if (CanBindLoopback(port)) return port;
        }

        var listener = new TcpListener(IPAddress.Loopback, 0);
        try
        {
            listener.Start();
            return ((IPEndPoint)listener.LocalEndpoint).Port;
        }
        finally
        {
            listener.Stop();
        }
    }

    private static bool CanBindLoopback(int port)
    {
        var listener = new TcpListener(IPAddress.Loopback, port);
        try
        {
            listener.Start();
            return true;
        }
        catch (SocketException)
        {
            return false;
        }
        finally
        {
            listener.Stop();
        }
    }

    private static Uri? ExtractPanelUri(string output)
    {
        foreach (var line in output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
        {
            if (!Uri.TryCreate(line.Trim(), UriKind.Absolute, out var uri)) continue;
            if (uri.Scheme != Uri.UriSchemeHttp || uri.Host is not ("127.0.0.1" or "localhost")) continue;
            return uri;
        }
        return null;
    }

    private async Task AppendLauncherLogAsync(string message)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(launcherLogPath)!);
            var entry = $"[{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss zzz}] {message}{Environment.NewLine}";
            await File.AppendAllTextAsync(launcherLogPath, entry, Encoding.UTF8);
        }
        catch
        {
            // Logging must never prevent the desktop shell from starting.
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
                using var response = await http.GetAsync(panelUri);
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
        webView.Source = panelUri;
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
        if (!ownsPanelService) return;
        try
        {
            using var panel = Process.Start(ServiceCommand("panel_service.py", "stop"));
            panel?.WaitForExit(5000);
            ownsPanelService = false;
        }
        catch
        {
            // The OS will reclaim the desktop process; service logs retain diagnostics.
        }
    }
}
