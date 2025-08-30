# 鸟类识别应用部署与运行指南 (阿里云版)

本项目是一个基于阿里云函数计算（FC）、对象存储（OSS）和表格存储（OTS）的鸟类识别应用。应用通过一个Flask Web服务器提供API，可以上传图片、进行鸟类识别、存储和检索识别结果。

## 1. 先决条件

在开始之前，请确保您已拥有：

- 一个有效的阿里云账户。
- 已开通函数计算（FC）、对象存储（OSS）、表格存储（OTS）和容器镜像服务（ACR）的访问权限。
- 一个具备`AliyunFCFullAccess`、`AliyunOSSFullAccess`、`AliyunOTSFullAccess`和`AliyunACRFullAccess`权限的RAM用户或角色，并获取其AccessKey ID和AccessKey Secret。
- 本地已安装并配置好Docker环境。
- 本地已安装并配置好阿里云CLI（`aliyun-cli`），并已通过`aliyun configure`配置好您的AccessKey和默认地域。

## 2. 阿里云服务设置

### 2.1. 创建对象存储（OSS）Bucket

1. 登录阿里云OSS控制台。
2. 创建一个新的存储空间（Bucket），例如`bird-detection-bucket`。请确保Bucket的读写权限设置为“私有”。
3. 记录下您的Bucket名称和所在的地域（Region），例如`cn-hangzhou`。

### 2.2. 创建表格存储（OTS）实例和数据表

1. 登录阿里云表格存储控制台。
2. 创建一个新的实例，例如`bird-detection-instance`。
3. 在创建的实例中，创建一个新的数据表（Table），例如`bird_detection_table`。
4. 定义主键（Primary Key）：
   - 主键名：`fileKey`
   - 主键类型：`STRING`
5. 记录下您的实例名称和数据表名称。

### 2.3. 创建容器镜像服务（ACR）命名空间和仓库

1. 登录阿里云容器镜像服务控制台。
2. 如果是首次使用，请设置您的注册表密码。
3. 创建一个新的命名空间，例如`bird-detection-ns`。
4. 在创建的命名空间下，创建一个新的镜像仓库，例如`bird-detection-repo`。
5. 记录下您的ACR仓库地址，例如`registry.cn-hangzhou.aliyuncs.com/bird-detection-ns/bird-detection-repo`。

## 3. 应用配置

1. 将项目根目录下的`secret.env.example`文件复制一份并重命名为`secret.env`。
2. 编辑`secret.env`文件，填入您在上一步中创建的阿里云服务的相关信息以及您的AccessKey：

```env
# Aliyun secrets
ALIBABA_CLOUD_ACCESS_KEY_ID=your_access_key_id
ALIBABA_CLOUD_ACCESS_KEY_SECRET=your_access_key_secret

# Aliyun service configurations
REGION=cn-hangzhou
BUCKET=your_bucket_name
OTS_INSTANCE=your_ots_instance_name
OTS_TABLE=your_ots_table_name

# ... (其他配置保持默认即可)
```

## 4. 构建和推送Docker镜像

- **应用配置**: 创建一个 `secret.env` 文件，参照 `secret.env.example` 进行配置。

### 3. 构建并推送镜像

```bash
# 登录到你的阿里云容器镜像服务
docker login --username=<your-acr-ee-username> registry.cn-hangzhou.aliyuncs.com

# 构建镜像
docker build -t registry.cn-hangzhou.aliyuncs.com/<your-namespace>/bird-detection:latest .

# 推送镜像
docker push registry.cn-hangzhou.aliyuncs.com/<your-namespace>/bird-detection:latest
```

### 4. 部署到函数计算

1.  **创建函数**:
    *   进入函数计算控制台，选择你的服务。
    *   创建一个新的函数。
    *   选择“使用容器镜像创建”。
    *   **镜像地址**: 选择你刚刚推送的镜像。
    *   **启动命令**: 填写 `/code/bootstrap`。
    *   **监听端口**: 填写 `9000`。
    *   **环境变量**: 点击“从文件导入”，上传你的 `secret.env` 文件。
    *   **执行超时**: 建议设置为 600 秒或更长，因为模型下载和首次推理可能耗时较长。
    *   **内存规格**: 建议选择 4GB 或更高。
    *   **网络配置**: 确保函数可以访问公网，以便下载模型和访问其他云服务。

2.  **配置触发器**:
    *   为函数创建一个 HTTP 触发器，允许匿名访问。

### 5. 使用 Web 应用

部署成功后，你可以通过 HTTP 触发器提供的 URL 访问 Web 应用。

1.  **注册和登录**:
    *   首次访问时，你需要注册一个新账户。
    *   注册后，使用你的用户名和密码登录。

2.  **功能**:
    *   **主页**: 登录后，你会看到一个文件列表，显示已处理的图像。
    *   **上传文件**: 你可以通过上传表单提交新的图像进行分析。
    *   **按物种搜索**: 在搜索框中输入物种名称，可以筛选出包含该物种的图像。
    *   **删除文件**: 每个文件旁边都有一个删除按钮，可以从系统中删除该文件及其相关数据。

### 6. 用户认证配置

应用包含用户认证功能，需要进行以下配置：

#### 6.1. 环境变量配置
在您的`secret.env`文件中添加以下配置：

```env
# Flask会话密钥（请生成一个随机字符串）
SECRET_KEY=your-secret-key-here
```

#### 6.2. 用户数据存储
用户账户存储在阿里云表格存储（OTS）中，需要创建用户表：

1. 在您的OTS实例中创建一个名为`users`的数据表
2. 设置主键：
   - 主键名：`username`
   - 主键类型：`STRING`
3. 用户密码使用bcrypt进行哈希存储

### 7. API 端点

**注意：** 所有API端点都需要用户认证。用户必须先登录才能访问任何功能。

#### 7.1. 认证端点
*   `GET /`: 主应用页面（未认证时重定向到登录页面）
*   `GET /login`: 用户登录页面
*   `POST /login`: 处理登录凭据
*   `GET /register`: 用户注册页面
*   `POST /register`: 处理用户注册
*   `GET /logout`: 注销当前用户

#### 7.2. 文件管理端点（需要认证）
*   `GET /health`: 健康检查
*   `POST /initialize`: 初始化函数（下载模型等）
*   `POST /invoke`: （供 OSS 触发器使用）处理 OSS 事件
*   `GET /files`: 列出已处理的文件
*   `POST /search`: 根据物种及其最小数量进行高级搜索
*   `GET /by-species`: 根据物种名称查询文件
*   `POST /reverse-thumb`: 根据缩略图反向查找文件
*   `POST /intersect`: 查找与给定对象有相同物种交集的文件
*   `POST /tags:update`: 批量更新文件的标签
*   `DELETE /files`: 删除文件

---

- 例如：
   ```bash
   docker push registry.cn-hangzhou.aliyuncs.com/bird-detection-ns/bird-detection-repo:latest
   ```

## 5. 部署到函数计算（FC）

1. 登录阿里云函数计算控制台。
2. 创建一个新的服务（Service）。
3. 在创建的服务中，创建一个新的函数（Function）。
4. **配置函数**：
   - **创建方式**：选择"使用容器镜像创建"。
   - **镜像地址**：选择"ACR个人版实例"，然后选择您在第4步中推送的镜像和标签。
   - **启动命令**：`/code/bootstrap`
   - **监听端口**：`9000`
   - **运行环境**：将`secret.env`文件中的所有键值对，作为环境变量添加到函数的配置中。
   - **触发器**：创建一个HTTP触发器，以便通过公网访问您的API。

5. **部署函数**：
   - 点击"创建"并等待函数部署完成。
   - 部署成功后，您将在HTTP触发器配置页面找到公网访问地址。

## 6. 使用API

您可以使用任何HTTP客户端（如`curl`、Postman或浏览器）与部署的应用进行交互。以下是一些主要的API端点：

- `GET /health`：检查应用健康状态。
- `POST /invoke`：触发对OSS中新上传文件的处理（由OSS触发器自动调用）。
- `GET /files`：列出已处理的文件。
- `GET /search`：根据文件名或元数据进行搜索。
- `DELETE /files`：删除文件及其相关数据。

详细的API使用方法请参考`app.py`中的代码。

## 8. 阿里云部署和运行指南

### 8.1. 部署前准备

#### 8.1.1. 阿里云账户和权限
1. 确保您有阿里云账户并已完成实名认证
2. 开通以下阿里云服务：
   - 对象存储OSS
   - 表格存储OTS
   - 容器镜像服务ACR
   - 函数计算FC
3. 创建RAM用户并授予相应权限：
   - `AliyunOSSFullAccess`
   - `AliyunOTSFullAccess`
   - `AliyunContainerRegistryFullAccess`
   - `AliyunFCFullAccess`

#### 8.1.2. 本地环境准备
1. 安装Docker Desktop
2. 安装阿里云CLI工具
3. 配置阿里云CLI凭证：
   ```bash
   aliyun configure
   ```

### 8.2. 服务配置步骤

#### 8.2.1. OSS配置
1. 创建OSS Bucket：
   ```bash
   # 替换为您的bucket名称和地域
   aliyun oss mb oss://your-bucket-name --region cn-hangzhou
   ```
2. 配置CORS规则（支持Web上传）
3. 创建触发器目录结构：
   - `uploads/` - 用户上传目录
   - `processed/` - 处理后文件目录

#### 8.2.2. OTS配置
1. 创建OTS实例：
   ```bash
   # 通过控制台或CLI创建实例
   aliyun ots CreateInstance --InstanceName your-instance-name --ClusterType SSD
   ```
2. 创建数据表：
   - `files` 表（主键：filename）
   - `users` 表（主键：username）

#### 8.2.3. ACR配置
1. 创建命名空间：
   ```bash
   aliyun cr CreateNamespace --NamespaceName your-namespace
   ```
2. 创建镜像仓库：
   ```bash
   aliyun cr CreateRepo --RepoNamespace your-namespace --RepoName bird-detection
   ```

### 8.3. 应用部署流程

#### 8.3.1. 构建和推送Docker镜像
1. 登录ACR：
   ```bash
   docker login --username=your-username registry.cn-hangzhou.aliyuncs.com
   ```
2. 构建镜像：
   ```bash
   docker build -t registry.cn-hangzhou.aliyuncs.com/your-namespace/bird-detection:latest .
   ```
3. 推送镜像：
   ```bash
   docker push registry.cn-hangzhou.aliyuncs.com/your-namespace/bird-detection:latest
   ```

#### 8.3.2. 函数计算部署
1. 创建函数计算服务：
   ```bash
   aliyun fc CreateService --ServiceName bird-detection-service
   ```
2. 创建函数：
   ```bash
   aliyun fc CreateFunction \
     --ServiceName bird-detection-service \
     --FunctionName bird-detection \
     --Runtime custom-container \
     --CodeUri registry.cn-hangzhou.aliyuncs.com/your-namespace/bird-detection:latest
   ```
3. 配置环境变量（通过控制台或CLI）
4. 设置内存和超时时间（建议3008MB内存，900秒超时）

#### 8.3.3. OSS触发器配置
1. 创建OSS触发器：
   ```bash
   aliyun fc CreateTrigger \
     --ServiceName bird-detection-service \
     --FunctionName bird-detection \
     --TriggerName oss-trigger \
     --TriggerType oss \
     --TriggerConfig '{"events":["oss:ObjectCreated:*"],"filter":{"key":{"prefix":"uploads/","suffix":".jpg"}}}'
   ```

### 8.4. 运行和监控

#### 8.4.1. 应用访问
1. 获取函数计算HTTP触发器URL
2. 配置自定义域名（可选）
3. 访问应用进行用户注册和登录

#### 8.4.2. 监控和日志
1. 通过函数计算控制台查看执行日志
2. 配置日志服务SLS进行日志分析
3. 设置云监控告警规则

#### 8.4.3. 性能优化
1. 根据使用情况调整函数内存配置
2. 配置预留实例减少冷启动时间
3. 使用CDN加速静态资源访问

### 8.5. 故障排除

#### 8.5.1. 常见问题
1. **镜像拉取失败**：检查ACR权限和网络连接
2. **函数超时**：增加超时时间或优化代码性能
3. **OSS访问失败**：检查RAM权限和Bucket配置
4. **OTS连接失败**：检查实例状态和网络配置

#### 8.5.2. 日志分析
1. 查看函数计算执行日志
2. 检查应用错误日志
3. 监控系统资源使用情况

---

## 总结

这个应用提供了一个完整的鸟类检测和管理系统，利用阿里云的强大基础设施来处理图像上传、AI 推理和数据存储。通过 Web 界面，用户可以轻松上传图像、查看检测结果，并管理他们的文件。应用包含完整的用户认证系统，确保数据安全和访问控制。

至此，您的鸟类识别应用已成功部署在阿里云上。#   x i n x i n  
 