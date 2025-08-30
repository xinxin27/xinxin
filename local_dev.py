#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地开发模式启动脚本
这个脚本允许应用在没有阿里云服务的情况下运行，用于本地开发和测试
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from typing import Dict, List, Optional

# 设置本地开发环境变量
os.environ['LOCAL_DEV_MODE'] = 'true'
os.environ['SECRET_KEY'] = 'local-dev-secret-key'
os.environ['BUCKET'] = 'local-dev-bucket'
os.environ['OTS_INSTANCE'] = 'local-dev-instance'
os.environ['OTS_REGION'] = 'cn-hangzhou'
os.environ['OTS_TABLE'] = 'BirdFiles'
os.environ['CONFIDENCE'] = '0.5'
os.environ['FC_SERVER_PORT'] = '9000'

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'local-dev-secret-key')

# Flask-Login 配置
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 本地用户存储（仅用于开发）
local_users = {}
local_files = [
    {
        'filename': 'sample_bird1.jpg',
        'upload_time': '2025-08-30 10:00:00',
        'species_counts': {'麻雀': 2, '燕子': 1},
        'thumb_url': '/static/sample_thumb1.jpg',
        'preview_url': '/static/sample_preview1.jpg'
    },
    {
        'filename': 'sample_bird2.jpg', 
        'upload_time': '2025-08-30 11:00:00',
        'species_counts': {'鸽子': 3, '乌鸦': 1},
        'thumb_url': '/static/sample_thumb2.jpg',
        'preview_url': '/static/sample_preview2.jpg'
    }
]

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    @staticmethod
    def get(user_id):
        return local_users.get(user_id)

    @staticmethod
    def create(username, password_hash):
        user = User(username, username, password_hash)
        local_users[username] = user
        return user

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/')
@login_required
def index():
    return render_template('index.html', files=local_files)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = local_users.get(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in local_users:
            flash('用户名已存在')
        else:
            password_hash = generate_password_hash(password)
            User.create(username, password_hash)
            flash('注册成功，请登录')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/files', methods=['GET'])
@login_required
def list_files():
    return jsonify({
        'status': 'success',
        'files': local_files,
        'total': len(local_files)
    })

@app.route('/search', methods=['POST'])
@login_required
def search_files():
    data = request.get_json() or {}
    species = data.get('species', '')
    min_count = data.get('min_count', 1)
    
    filtered_files = []
    for file in local_files:
        if species in file['species_counts'] and file['species_counts'][species] >= min_count:
            filtered_files.append(file)
    
    return jsonify({
        'status': 'success',
        'files': filtered_files,
        'total': len(filtered_files)
    })

@app.route('/search_ui', methods=['GET'])
@login_required
def search_by_species_ui():
    species = request.args.get('species', '')
    min_count = int(request.args.get('min_count', 1))
    
    filtered_files = local_files
    if species:
        filtered_files = []
        for file in local_files:
            if species in file['species_counts'] and file['species_counts'][species] >= min_count:
                filtered_files.append(file)
    
    return render_template('index.html', files=filtered_files, search_species=species, search_min_count=min_count)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('没有选择文件')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('没有选择文件')
            return redirect(url_for('index'))
        
        if file:
            # 在本地开发模式中，我们只是模拟上传成功
            flash(f'文件 {file.filename} 上传成功（本地开发模式）')
            return redirect(url_for('index'))
    
    return render_template('upload.html')

@app.route('/delete_file', methods=['POST'])
@login_required
def delete_file_ui():
    file_key = request.form.get('fileKey')
    if file_key:
        # 在本地开发模式中，我们只是模拟删除成功
        flash(f'文件 {file_key} 删除成功（本地开发模式）')
    else:
        flash('删除失败：未指定文件')
    return redirect(url_for('index'))

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'mode': 'local_development',
        'message': '本地开发模式运行正常'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 本地开发模式启动")
    print("="*60)
    print("📝 这是一个简化的本地开发版本，不需要阿里云服务")
    print("🔧 功能包括：用户注册/登录、文件列表、搜索等")
    print("🌐 访问地址: http://127.0.0.1:9000")
    print("📋 测试账户: 可以注册新用户或使用任意用户名/密码")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=9000, debug=True)