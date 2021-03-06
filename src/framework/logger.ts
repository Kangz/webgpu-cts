import { IParamsSpec } from './params/index.js';
import { getStackTrace, now } from './util.js';
import { version } from './version.js';

type Status = 'running' | 'pass' | 'warn' | 'fail';
interface ITestLog {
  path: string;
  cases: IResult[];
}
export interface IResult {
  name: string;
  params: IParamsSpec | null;
  status: Status;
  logs?: string[];
  timems: number;
}

export class Logger {
  readonly results: ITestLog[] = [];

  constructor() {}

  record(path: string): [ITestLog, GroupRecorder] {
    const cases: IResult[] = [];
    const test: ITestLog = { path, cases };
    this.results.push(test);
    return [test, new GroupRecorder(test)];
  }

  asJSON(space?: number): string {
    return JSON.stringify({ version, results: this.results }, undefined, space);
  }
}

export class GroupRecorder {
  private test: ITestLog;

  constructor(test: ITestLog) {
    this.test = test;
  }

  record(name: string, params: IParamsSpec | null): [IResult, CaseRecorder] {
    const result: IResult = { name, params, status: 'running', timems: -1 };
    this.test.cases.push(result);
    return [result, new CaseRecorder(result)];
  }
}

export class CaseRecorder {
  private result: IResult;
  private failed = false;
  private warned = false;
  private startTime = -1;
  private logs: string[] = [];

  constructor(result: IResult) {
    this.result = result;
  }

  start() {
    this.startTime = now();
    this.logs = [];
    this.failed = false;
    this.warned = false;
  }

  finish() {
    if (this.startTime < 0) {
      throw new Error('finish() before start()');
    }
    const endTime = now();
    this.result.timems = endTime - this.startTime;
    this.result.status = this.failed ? 'fail' : this.warned ? 'warn' : 'pass';

    this.result.logs = this.logs;
  }

  log(msg: string) {
    this.logs.push(msg);
  }

  warn(msg?: string) {
    this.warned = true;
    let m = 'WARN';
    if (msg) {
      m += ': ' + msg;
    }
    m += ' ' + getStackTrace(new Error());
    this.log(m);
  }

  fail(msg?: string) {
    this.failed = true;
    let m = 'FAIL';
    if (msg) {
      m += ': ' + msg;
    }
    m += ' ' + getStackTrace(new Error());
    this.log(m);
  }

  threw(e: Error) {
    this.failed = true;
    let m = 'EXCEPTION';
    m += ' ' + getStackTrace(e);
    this.log(m);
  }
}
