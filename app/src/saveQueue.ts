/** Serialize autosaves so a slower old write cannot overwrite a newer edit. */
export class SaveQueue {
  private pending = new Map<string, () => Promise<void>>()
  private running: Promise<void> | null = null
  private timer: ReturnType<typeof setTimeout> | null = null

  constructor(private onError: (error: unknown) => void, private delay = 380) {}

  schedule(key: string, write: () => Promise<void>): void {
    this.pending.set(key, write)
    if (this.timer) clearTimeout(this.timer)
    this.timer = setTimeout(() => { void this.flush().catch(this.onError) }, this.delay)
  }

  async flush(): Promise<void> {
    if (this.timer) { clearTimeout(this.timer); this.timer = null }
    if (this.running) await this.running
    if (!this.pending.size) return
    this.running = this.drain()
    try { await this.running } finally { this.running = null }
    // An edit may have arrived as the previous drain settled.
    if (this.pending.size) await this.flush()
  }

  private async drain(): Promise<void> {
    while (this.pending.size) {
      const [key, write] = this.pending.entries().next().value!
      this.pending.delete(key)
      try { await write() } catch (error) {
        // Keep the latest local edit available for a retry.
        if (!this.pending.has(key)) this.pending.set(key, write)
        throw error
      }
    }
  }
}
