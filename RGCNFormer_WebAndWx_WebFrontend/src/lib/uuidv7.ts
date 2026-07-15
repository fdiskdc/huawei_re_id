/**
 * uuidv7.ts - 零依赖 UUID v7 生成器 / Zero-dependency UUID v7 generator
 *
 * 符合 RFC 9562 的 UUID v7:48 位 Unix 毫秒时间戳 + version bits + 随机位。结果
 * 时间可排序,且有密码学强度的随机性。 / RFC 9562 compliant UUID v7: 48-bit
 * Unix ms timestamp + version bits + random bits. Time-sortable with
 * cryptographically random uniqueness.
 *
 * 功能模块 / Modules:
 * - uuidv7(): 生成 UUID v7 字符串 / Generate a UUID v7 string
 *
 * 输入 / Inputs:
 * - 无 / None
 *
 * 输出 / Outputs:
 * - string - 形如 "01890c5f-...-...-...-..." 的 UUID / UUID string
 *
 * 数据流 / Data Flow:
 * 1. 取 Date.now() → 48-bit 时间戳 / Get 48-bit timestamp
 * 2. crypto.getRandomValues 10 字节随机 / 10 random bytes
 * 3. 按 RFC 9562 拼接 hex 字符串 / Concatenate per RFC 9562
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: Date.now()、crypto.getRandomValues (Web Crypto API)
 * - 被调用 / Called by: src/lib/api.ts(生成 jobId)、其他需要唯一 ID 的模块
 *
 * 使用示例 / Usage Example:
 *     import { uuidv7 } from '@/lib/uuidv7';
 *     const id = uuidv7();
 *     console.log(id);  // "01890c5f-..."
 *
 * 作者 / Author: 项目组 / Project Team
 * 版本 / Version: 1.0
 */
/**
 * Zero-dependency UUID v7 generator (RFC 9562)
 * Format: 48-bit Unix ms timestamp + version bits + random bits
 * Result is time-sortable with cryptographically random uniqueness
 */

export function uuidv7(): string {
  const timestamp = Date.now();
  const random = crypto.getRandomValues(new Uint8Array(10));

  // Build hex string: 12 hex chars (48-bit timestamp) + 4 hex chars (version + rand) + ...
  const hex = [
    // Bytes 0-5: 48-bit big-endian timestamp
    (timestamp >> 40) & 0xff,
    (timestamp >> 32) & 0xff,
    (timestamp >> 24) & 0xff,
    (timestamp >> 16) & 0xff,
    (timestamp >> 8) & 0xff,
    timestamp & 0xff,
    // Byte 6: version 7 in top 4 bits, rand in lower
    (0x7 << 4) | (random[0] & 0x0f),
    // Byte 7: variant 0b10 in top 2 bits, rand in lower
    0x80 | (random[1] & 0x3f),
    // Bytes 8-15: remaining random
    ...Array.from(random.slice(2)),
  ]
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');

  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
}
