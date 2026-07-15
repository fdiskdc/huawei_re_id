"""
中文：微信小程序 API 端点。提供用户登录和会话管理。
English: WeChat Mini Program API endpoints. Provides user login and session management.
"""
import json
import requests as http_requests
from flask import Blueprint, jsonify, request, current_app

from mrmodn_backend.core.config import config, get_logger

wechat_bp = Blueprint('wechat', __name__)
logger = get_logger('api.wechat')


@wechat_bp.route('/mrmodn/api/v1/wx/login', methods=['POST'])
def wx_login():
    """
    中文：微信小程序登录端点。接收 loginCode，调用微信 API 换取 openid，并在 Redis 中存储/更新用户信息。
    English: WeChat Mini Program login endpoint. Receives loginCode, exchanges it for openid via WeChat API, and stores/updates user info in Redis.
    """
    # Get data from request
    data = request.get_json()

    # Validate input
    if not data:
        return jsonify({"error": "No data provided"}), 400

    login_code = data.get('loginCode')
    nickname = data.get('nickname')
    avatar_url = data.get('avatarUrl')

    if not login_code:
        return jsonify({"error": "loginCode is required"}), 400

    # Check if WeChat app credentials are configured
    if not config.WX_APPID or not config.WX_SECRET:
        logger.error("WeChat app credentials not configured")
        return jsonify({
            "error": "WeChat app credentials not configured",
            "detail": "Please set WX_APPID and WX_SECRET environment variables"
        }), 500

    logger.info(f"Processing WeChat login request for code: {login_code[:10]}...")

    try:
        # Call WeChat API to exchange code for openid and session_key
        params = {
            'appid': config.WX_APPID,
            'secret': config.WX_SECRET,
            'js_code': login_code,
            'grant_type': 'authorization_code'
        }

        response = http_requests.get(config.WX_LOGIN_URL, params=params, timeout=10)
        response_data = response.json()

        # Check if WeChat API returned an error
        if 'errcode' in response_data:
            error_msg = response_data.get('errmsg', 'Unknown WeChat API error')
            logger.error(f"WeChat API error: {response_data.get('errcode')} - {error_msg}")
            return jsonify({
                "error": f"WeChat API error: {response_data.get('errcode')}",
                "detail": error_msg
            }), 400

        # Extract openid and session_key
        openid = response_data.get('openid')
        session_key = response_data.get('session_key')

        if not openid:
            logger.error("WeChat API did not return openid")
            return jsonify({
                "error": "Failed to get openid from WeChat API",
                "detail": response_data
            }), 500

        # Check if user already exists in Redis
        redis_client = current_app.config.get('REDIS_CLIENT')
        user = {
            'openid': openid,
            'session_key': session_key,
            'nickname': nickname,
            'avatarUrl': avatar_url
        }

        if redis_client:
            try:
                user_key = f"wx_user:{openid}"
                user_data = redis_client.get(user_key)

                if user_data:
                    existing_user = json.loads(user_data)
                    logger.info(f"Existing user found: {openid}")
                    user['session_key'] = session_key
                    if nickname is not None:
                        user['nickname'] = nickname
                    else:
                        user['nickname'] = existing_user.get('nickname')
                    if avatar_url is not None:
                        user['avatarUrl'] = avatar_url
                    else:
                        user['avatarUrl'] = existing_user.get('avatarUrl')
                    redis_client.setex(
                        user_key,
                        30 * 24 * 3600,  # 30 days in seconds
                        json.dumps(user, ensure_ascii=False)
                    )
                else:
                    # Store new user data in Redis with 30 days TTL
                    redis_client.setex(
                        user_key,
                        30 * 24 * 3600,  # 30 days in seconds
                        json.dumps(user, ensure_ascii=False)
                    )
                    logger.info(f"New user created: {openid}")
            except Exception as e:
                logger.error(f"Redis error during user storage: {e}")
                # Continue with user object even if Redis fails

        # Prepare user info for response (exclude session_key for security)
        user_info = {
            'openid': user['openid'],
            'nickname': user.get('nickname'),
            'avatarUrl': user.get('avatarUrl')
        }

        # Return success response with user info
        return jsonify({
            "code": 0,
            "openid": openid,
            "data": user_info,
            "message": "Login successful"
        }), 200

    except http_requests.exceptions.Timeout:
        logger.error("WeChat API request timeout")
        return jsonify({
            "error": "WeChat API request timeout",
            "detail": "Failed to connect to WeChat server"
        }), 504
    except http_requests.exceptions.RequestException as e:
        logger.error(f"WeChat API request error: {e}")
        return jsonify({
            "error": "Failed to call WeChat API",
            "detail": str(e)
        }), 500
    except Exception as e:
        import traceback
        error_msg = f"WeChat login error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500
