export const description = `
Unit tests for TestGroup.
`;

import { DefaultFixture, Fixture, TestGroup, poptions } from '../../framework/index.js';
import { TestGroupTest } from './test_group_test.js';

export const g = new TestGroup(TestGroupTest);

g.test('default fixture', async t0 => {
  let seen = 0;
  function print(t: Fixture) {
    seen++;
  }

  const g = new TestGroup(DefaultFixture);

  g.test('test', print);
  g.test('testp', print).params([{ a: 1 }]);

  await t0.run(g);
  t0.expect(seen === 2);
});

g.test('custom fixture', async t0 => {
  let seen = 0;
  class Printer extends DefaultFixture {
    print() {
      seen++;
    }
  }

  const g = new TestGroup(Printer);

  g.test('test', t => {
    t.print();
  });
  g.test('testp', t => {
    t.print();
  }).params([{ a: 1 }]);

  await t0.run(g);
  t0.expect(seen === 2);
});

g.test('duplicate test name', t => {
  const g = new TestGroup(DefaultFixture);
  g.test('abc', () => {});

  t.shouldThrow(() => {
    g.test('abc', () => {});
  });
});

const badChars = Array.from('"`~@#$+=\\|!^&*[]<>{}-\'.,');
g.test('invalid test name', t => {
  const g = new TestGroup(DefaultFixture);

  t.shouldThrow(() => {
    g.test('a' + t.params.char + 'b', () => {});
  });
}).params(poptions('char', badChars));
