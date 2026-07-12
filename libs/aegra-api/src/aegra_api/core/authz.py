"""租户/用户归属过滤 —— 多租户隔离的单一权威入口。

所有面向请求的租户查询都必须用 owner_filter 生成 WHERE 条件,禁止在各处手写
user_id/tenant_id 过滤:漏一处即越权(参见 GHSA-m98r-6667-4wq7)。切换隔离语义
(复合隔离 ↔ tenant 内共享)只需改这一个函数,调用点不动。
"""

from typing import Any

from sqlalchemy import ColumnElement, and_, or_

from aegra_api.models.auth import User


def scope(model: Any, identity: str, tenant_id: str | None, *, allow_system: bool = False) -> ColumnElement[bool]:
    """底层归属过滤:直接以 identity + tenant_id 生成 WHERE 条件。

    供只持有字符串标识(而非 User 对象)的 service 层使用。tenant_id 用
    IS NOT DISTINCT FROM 匹配:为 None 时匹配 tenant_id 为 NULL 的资源,有值时
    只匹配相等的资源(tenant 硬隔离边界)。allow_system=True 额外放行
    user_id == "system" 的共享资源(如系统预置 assistant);写操作应保持 False。
    """
    own = and_(
        model.user_id == identity,
        model.tenant_id.is_not_distinct_from(tenant_id),
    )
    if allow_system:
        return or_(own, model.user_id == "system")
    return own


def owner_filter(model: Any, user: User, *, allow_system: bool = False) -> ColumnElement[bool]:
    """scope 的 User 对象包装,供持有 User 的 api 层使用。"""
    return scope(model, user.identity, getattr(user, "tenant_id", None), allow_system=allow_system)


def owns(obj: Any, user: User, *, allow_system: bool = False) -> bool:
    """Python 层归属判断,语义与 owner_filter 一致(用于已取出的 ORM 实例)。

    tenant 用 == 比较:Python 中 None == None 为 True,等价于 SQL 的
    IS NOT DISTINCT FROM。用于"先按 id 取出、再判归属"的场景(如 thread 可选存在)。
    """
    if obj.user_id == user.identity and getattr(obj, "tenant_id", None) == getattr(user, "tenant_id", None):
        return True
    return bool(allow_system and obj.user_id == "system")
