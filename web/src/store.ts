// Web budget and visitor limits in SQLite (a file on the host's volume).
//
// Budget model A (user, 2026-11-09): every issued token reserves its seconds (5 minutes) at
// once; nothing is credited back when a call is shorter. So the daily 20 minutes allow at
// most four tokens, and the monthly 60 minutes twelve. The agent counts real durations as a
// backstop (agent/demo.py). A visitor (keyed hash of the address) gets a few tokens a day;
// the visitor value is erased the next day and only the minutes stay for the month.

import { DatabaseSync } from "node:sqlite";

export interface Limits {
  webMinutesPerDay: number;
  webMinutesPerMonth: number;
  callSeconds: number;
  visitorTokensPerDay: number;
}

export type Refusal = "visitor" | "daily" | "monthly";
export type Reservation = { ok: true; id: number; seconds: number } | { ok: false; reason: Refusal };

const MIN_SECONDS = 60; // less than this left: no token (same rule as the agent)
const dayFormat = new Intl.DateTimeFormat("sv-SE", {
  timeZone: "Europe/Berlin",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
});

/** The calendar day in Europe/Berlin as YYYY-MM-DD. */
export function berlinDay(now: Date): string {
  return dayFormat.format(now);
}

export class UsageStore {
  private db: DatabaseSync;
  private limits: Limits;

  constructor(path: string, limits: Limits) {
    this.limits = limits;
    this.db = new DatabaseSync(path);
    this.db.exec("PRAGMA busy_timeout = 10000");
    this.db.exec("PRAGMA journal_mode = WAL");
    this.db.exec(
      "CREATE TABLE IF NOT EXISTS reservations (" +
        "id INTEGER PRIMARY KEY, day TEXT NOT NULL, visitor TEXT, seconds INTEGER NOT NULL)",
    );
  }

  reserve(visitor: string, now: Date): Reservation {
    const day = berlinDay(now);
    const lim = this.limits;
    this.db.exec("BEGIN IMMEDIATE"); // one decision at a time, across connections
    try {
      // Yesterday's visitors are forgotten; the minutes stay for the month.
      this.db.prepare("UPDATE reservations SET visitor = NULL WHERE day < ? AND visitor IS NOT NULL").run(day);
      const mine = this.num("SELECT count(*) AS n FROM reservations WHERE day = ? AND visitor = ?", day, visitor);
      const usedDay = this.num("SELECT coalesce(sum(seconds), 0) AS n FROM reservations WHERE day = ?", day);
      const usedMonth = this.num(
        "SELECT coalesce(sum(seconds), 0) AS n FROM reservations WHERE substr(day, 1, 7) = ?",
        day.slice(0, 7),
      );
      const leftMonth = lim.webMinutesPerMonth * 60 - usedMonth;
      const leftDay = lim.webMinutesPerDay * 60 - usedDay;
      let result: Reservation;
      if (mine >= lim.visitorTokensPerDay) result = { ok: false, reason: "visitor" };
      else if (leftMonth < MIN_SECONDS) result = { ok: false, reason: "monthly" };
      else if (leftDay < MIN_SECONDS) result = { ok: false, reason: "daily" };
      else {
        const seconds = Math.min(lim.callSeconds, leftDay, leftMonth);
        const run = this.db
          .prepare("INSERT INTO reservations (day, visitor, seconds) VALUES (?, ?, ?)")
          .run(day, visitor, seconds);
        result = { ok: true, id: Number(run.lastInsertRowid), seconds };
      }
      this.db.exec("COMMIT");
      return result;
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }

  /** Takes a reservation back (a token that could not be signed). */
  release(id: number): void {
    this.db.prepare("DELETE FROM reservations WHERE id = ?").run(id);
  }

  close(): void {
    this.db.close();
  }

  private num(sql: string, ...params: string[]): number {
    const row = this.db.prepare(sql).get(...params) as { n: number };
    return Number(row.n);
  }
}
