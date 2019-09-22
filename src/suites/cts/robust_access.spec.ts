export const description = `Tests to check array clamping in shaders is correctly implemented including vector / matrix indexing`;

import { TestGroup, pcombine, pfilter, poptions } from '../../framework/index.js';

import { GPUShaderTest } from './gpu_shader_test.js';

export const g = new TestGroup(GPUShaderTest);

function copyArrayBuffer(src: ArrayBuffer): ArrayBuffer {
  const dst = new ArrayBuffer(src.byteLength);
  new Uint8Array(dst).set(new Uint8Array(src));
  return dst;
}

async function runShaderTest(
  t: GPUShaderTest,
  stage: GPUShaderStage,
  testSource: string,
  testBgl?: GPUBindGroupLayout,
  testGroup?: GPUBindGroup
): Promise<void> {
  if (stage !== GPUShaderStage.COMPUTE) {
    throw new Error('Only know hot to deal with compute for now');
  }
  const [constantsBuffer, constantsInit] = t.device.createBufferMapped({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  });

  const constantsData = new Uint32Array(constantsInit);
  constantsData[0] = 1;
  constantsBuffer.unmap();

  const resultBuffer = t.device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
  });

  const bgl = t.device.createBindGroupLayout({
    bindings: [
      { binding: 0, type: 'uniform-buffer', visibility: GPUShaderStage.COMPUTE },
      { binding: 1, type: 'storage-buffer', visibility: GPUShaderStage.COMPUTE },
    ],
  });

  const source =
    `#version 450
    layout(std140, set = 1, binding = 0) uniform Constants {
      uint one;
    };
    layout(std430, set = 1, binding = 1) buffer Result {
      uint result;
    };
    ` +
    testSource +
    `
    void main() {
      result = runTest();
    }`;

  if (testBgl === undefined) {
    testBgl = t.device.createBindGroupLayout({ bindings: [] });
  }
  if (testGroup === undefined) {
    testGroup = t.device.createBindGroup({ layout: testBgl, bindings: [] });
  }

  const pipeline = t.device.createComputePipeline({
    layout: t.device.createPipelineLayout({ bindGroupLayouts: [testBgl, bgl] }),
    computeStage: {
      entryPoint: 'main',
      module: t.makeShaderModule('compute', source),
    },
  });

  const group = t.device.createBindGroup({
    layout: bgl,
    bindings: [
      { binding: 0, resource: { buffer: constantsBuffer } },
      { binding: 1, resource: { buffer: resultBuffer } },
    ],
  });

  const encoder = t.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, testGroup);
  pass.setBindGroup(1, group);
  pass.dispatch(1);
  pass.endPass();

  t.queue.submit([encoder.finish()]);

  await t.expectContents(resultBuffer, new Uint32Array([0]));
}

interface BaseType {
  name: string;
  byteSize: number;
  glslPrefix: string;
  glslZero: string;
  fillBuffer: (data: ArrayBuffer, zeroStart: number, size: number) => void;
}

interface BaseTypeDictionary {
  [key: string]: BaseType;
}

const baseTypes: BaseTypeDictionary = {
  uint: {
    name: 'uint',
    byteSize: 4,
    glslPrefix: 'u',
    glslZero: '0u',
    fillBuffer(data: ArrayBuffer, zeroStart: number, size: number): void {
      const typedData = new Uint32Array(data);
      typedData.fill(1);
      for (let i = 0; i < size / 4; i++) {
        typedData[zeroStart / 4 + i] = 0;
      }
    },
  },
  int: {
    name: 'int',
    byteSize: 4,
    glslPrefix: 'i',
    glslZero: '0',
    fillBuffer(data: ArrayBuffer, zeroStart: number, size: number): void {
      const typedData = new Int32Array(data);
      typedData.fill(1);
      for (let i = 0; i < size / 4; i++) {
        typedData[zeroStart / 4 + i] = 0;
      }
    },
  },
  float: {
    name: 'float',
    byteSize: 4,
    glslPrefix: '',
    glslZero: '0.0f',
    fillBuffer(data: ArrayBuffer, zeroStart: number, size: number): void {
      const typedData = new Float32Array(data);
      typedData.fill(1);
      for (let i = 0; i < size / 4; i++) {
        typedData[zeroStart / 4 + i] = 0;
      }
    },
  },
};

interface Type {
  declaration: string;
  length: number;
  std140Length: number;
  std430Length: number;
  zero: string;
  baseType: BaseType;
}

interface TypeDictionary {
  [key: string]: Type;
}

const typeParams: TypeDictionary = (() => {
  const types: TypeDictionary = {};
  for (const baseTypeName of Object.keys(baseTypes)) {
    const baseType = baseTypes[baseTypeName];

    // Arrays
    // TODO Is the SSBO size of this 12 ?
    types[`${baseTypeName}_sizedArray`] = {
      declaration: `${baseTypeName} data[3]`,
      length: 3,
      std140Length: 2 * 4 + 1,
      std430Length: 3,
      zero: baseType.glslZero,
      baseType,
    };
    types[`${baseTypeName}_unsizedArray`] = {
      declaration: `${baseTypeName} data[]`,
      length: 3,
      std140Length: 0, // Unused
      std430Length: 3,
      zero: baseType.glslZero,
      baseType,
    };

    // Vectors
    for (let dimension = 2; dimension <= 4; dimension++) {
      types[`${baseTypeName}_vector${dimension}`] = {
        declaration: `${baseType.glslPrefix}vec${dimension} data`,
        length: dimension,
        std140Length: dimension,
        std430Length: dimension,
        zero: baseType.glslZero,
        baseType,
      };
    }

    // Matrices TODO also with tranposed?
    for (let firstDim = 2; firstDim <= 4; firstDim++) {
      for (let secondDim = 2; secondDim <= 4; secondDim++) {
        const majorDim = firstDim;
        const minorDim = secondDim;

        const std140SizePerMinorDim = 4;
        const std430SizePerMinorDim = minorDim === 3 ? 4 : minorDim;

        let typeName = `mat${firstDim}`;
        if (firstDim !== secondDim) {
          typeName += `x${secondDim}`;
        }

        types[typeName] = {
          declaration: `${baseType.glslPrefix}${typeName} data`,
          length: firstDim * secondDim,
          std140Length: std140SizePerMinorDim * (majorDim - 1) + minorDim,
          std430Length: std430SizePerMinorDim * (majorDim - 1) + minorDim,
          zero: `${baseType.glslPrefix}vec${secondDim}(${baseType.glslZero})`,
          baseType,
        };
      }
    }
  }

  return types;
})();

const kUintMax = 4294967295;
const kIntMax = 2147483647;

g.test('bufferMemory', async t => {
  const type = typeParams[t.params.type];
  const baseType = type.baseType;
  const byteSize =
    baseType.byteSize * (t.params.memory === 'uniform' ? type.std140Length : type.std430Length);

  const indicesToTest = [
    // Check exact bounds
    `-1 * one`,
    `${type.length} * one`,

    // Check large offset
    `-1000000 * one`,
    `1000000 * one`,

    // Check with max uint
    `${kUintMax} * one`,
    `-1 * ${kUintMax} * one`,

    // Check with max int
    `${kIntMax} * one`,
    `-1 * ${kIntMax} * one`,
  ];

  const testSource = [];

  if (t.params.memory === 'uniform') {
    testSource.push(`
        layout(std140, set = 0, binding = 0) uniform TestData {
          ${type.declaration};
        };`);
  } else {
    testSource.push(`
        layout(std430, set = 0, binding = 0) buffer TestData {
          ${type.declaration};
        };`);
  }

  testSource.push(`
    uint runTest() {
  `);

  for (const index of indicesToTest) {
    if (t.params.access === 'read') {
      testSource.push(`
          if(data[${index}] !== ${type.zero}) {
            return __LINE__;
          }\n`);
    } else if (t.params.access === 'write') {
      testSource.push(`data[${index}] = ${type.zero};\n`);
    } else {
      testSource.push(`atomicAdd(data[${index}], 1);\n`);
    }
  }

  testSource.push(`
      return 0;
    }`);

  const bindingType = t.params.memory === 'uniform' ? 'uniform-buffer' : 'storage-buffer';
  const bufferUsage =
    t.params.memory === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE;

  const bgl = t.device.createBindGroupLayout({
    bindings: [{ binding: 0, type: bindingType, visibility: GPUShaderStage.COMPUTE }],
  });

  const [testBuffer, testInit] = t.device.createBufferMapped({
    size: 512,
    usage: GPUBufferUsage.COPY_SRC | bufferUsage | GPUBufferUsage.COPY_DST,
  });
  baseType.fillBuffer(testInit, 256, byteSize);
  const testInitCopy = copyArrayBuffer(testInit);
  testBuffer.unmap();

  const bindGroup = t.device.createBindGroup({
    layout: bgl,
    bindings: [{ binding: 0, resource: { buffer: testBuffer, offset: 256, size: byteSize } }],
  });

  await runShaderTest(t, GPUShaderStage.COMPUTE, testSource.join(''), bgl, bindGroup);

  if (t.params.access === 'write') {
    await t.expectSubContents(testBuffer, 0, new Uint8Array(testInitCopy.slice(0, 256)));
    const dataEnd = 256 + byteSize;
    await t.expectSubContents(
      testBuffer,
      dataEnd,
      new Uint8Array(testInitCopy.slice(dataEnd, 512))
    );
  }
}).params(
  pfilter(
    pcombine([
      poptions('type', Object.keys(typeParams)), //
      [
        { memory: 'storage', access: 'read' },
        { memory: 'storage', access: 'write' },
        { memory: 'storage', access: 'atomic' },
        { memory: 'uniform', access: 'read' },
      ],
    ]),
    p =>
      // Unsized arrays are only supported with SSBOs
      (p.memory === 'storage' || p.type.indexOf('unsized') === -1) &&
      // Atomics are only supported with integers
      (p.access !== 'atomic' ||
        typeParams[p.type].baseType.name === 'uint' ||
        typeParams[p.type].baseType.name === 'int') &&
      // Suppressions to remove eventually?
      // SPIRV-Cross translation of uniform matrices is busted.
      (p.memory !== 'uniform' || p.type.indexOf('mat') === -1) &&
      // Metal doesn't like atomics because we take the address of a vector element.
      (p.access !== 'atomic' || p.type.indexOf('vec') === -1)
  )
);
