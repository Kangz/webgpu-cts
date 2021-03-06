import {
  DefaultFixture,
  TestGroup,
  Fixture,
  paramsEquals,
  ICaseID,
} from '../../framework/index.js';
import { Logger } from '../../framework/logger.js';

export class TestGroupTest extends DefaultFixture {
  async run<F extends Fixture>(g: TestGroup<F>): Promise<void> {
    const [, rec] = new Logger().record('');
    await Promise.all(Array.from(g.iterate(rec)).map(test => test.run()));
  }

  enumerate<F extends Fixture>(g: TestGroup<F>): ICaseID[] {
    const cases = [];
    const [, rec] = new Logger().record('');
    for (const test of g.iterate(rec)) {
      cases.push(test.testcase);
    }
    return cases;
  }

  expectCases<F extends Fixture>(g: TestGroup<F>, cases: ICaseID[]): void {
    const gcases = this.enumerate(g);

    if (this.expect(gcases.length === cases.length)) {
      for (let i = 0; i < cases.length; ++i) {
        this.expect(gcases[i].name === cases[i].name);
        this.expect(paramsEquals(gcases[i].params, cases[i].params));
      }
    }
  }
}
