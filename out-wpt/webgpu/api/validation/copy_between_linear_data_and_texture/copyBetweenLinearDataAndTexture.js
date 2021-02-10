/**
 * AUTO-GENERATED - DO NOT EDIT. Source: https://github.com/gpuweb/cts
 **/ import { poptions } from '../../../../common/framework/params_builder.js';
import { assert } from '../../../../common/framework/util/util.js';
import { kSizedTextureFormatInfo } from '../../../capability_info.js';
import { align } from '../../../util/math.js';
import { ValidationTest } from '../validation_test.js';

export const kAllTestMethods = ['WriteTexture', 'CopyBufferToTexture', 'CopyTextureToBuffer'];

export class CopyBetweenLinearDataAndTextureTest extends ValidationTest {
  bytesInACompleteRow(copyWidth, format) {
    const info = kSizedTextureFormatInfo[format];
    assert(copyWidth % info.blockWidth === 0);
    return (info.bytesPerBlock * copyWidth) / info.blockWidth;
  }

  /**
   * Validate a copy and compute the number of bytes it needs. If the copy is invalid, computes a
   * guess assuming `bytesPerRow` and `rowsPerImage` should be optimal.
   */
  dataBytesForCopy(layout, format, copyExtent, { method }) {
    var _layout$offset;
    const info = kSizedTextureFormatInfo[format];
    assert(copyExtent.width % info.blockWidth === 0);
    const widthInBlocks = copyExtent.width / info.blockWidth;
    assert(copyExtent.height % info.blockHeight === 0);
    const heightInBlocks = copyExtent.height / info.blockHeight;
    const bytesInLastRow = widthInBlocks * info.bytesPerBlock;

    let valid = true;
    const offset =
      (_layout$offset = layout.offset) !== null && _layout$offset !== void 0 ? _layout$offset : 0;
    if (method !== 'WriteTexture') {
      if (offset % info.bytesPerBlock !== 0) valid = false;
      if (layout.bytesPerRow && layout.bytesPerRow % 256 !== 0) valid = false;
    }
    if (layout.bytesPerRow !== undefined && bytesInLastRow > layout.bytesPerRow) valid = false;
    if (layout.rowsPerImage !== undefined && heightInBlocks > layout.rowsPerImage) valid = false;

    let requiredBytesInCopy = 0;
    {
      var _bytesPerRow, _rowsPerImage;
      let { bytesPerRow, rowsPerImage } = layout;

      // If heightInBlocks > 1, layout.bytesPerRow must be specified.
      if (heightInBlocks > 1 && bytesPerRow === undefined) valid = false;
      // If copyExtent.depth > 1, layout.bytesPerRow and layout.rowsPerImage must be specified.
      if (copyExtent.depth > 1 && rowsPerImage === undefined) valid = false;
      // If specified, layout.bytesPerRow must be greater than or equal to bytesInLastRow.
      if (bytesPerRow !== undefined && bytesPerRow < bytesInLastRow) valid = false;
      // If specified, layout.rowsPerImage must be greater than or equal to heightInBlocks.
      if (rowsPerImage !== undefined && rowsPerImage < heightInBlocks) valid = false;

      (_bytesPerRow = bytesPerRow) !== null && _bytesPerRow !== void 0
        ? _bytesPerRow
        : (bytesPerRow = align(info.bytesPerBlock * widthInBlocks, 256));
      (_rowsPerImage = rowsPerImage) !== null && _rowsPerImage !== void 0
        ? _rowsPerImage
        : (rowsPerImage = heightInBlocks);

      if (copyExtent.depth > 1) {
        const bytesPerImage = bytesPerRow * rowsPerImage;
        const bytesBeforeLastImage = bytesPerImage * (copyExtent.depth - 1);
        requiredBytesInCopy += bytesBeforeLastImage;
      }
      if (copyExtent.depth > 0) {
        if (heightInBlocks > 1) requiredBytesInCopy += bytesPerRow * (heightInBlocks - 1);
        if (heightInBlocks > 0) requiredBytesInCopy += bytesInLastRow;
      }
    }

    return { minDataSize: offset + requiredBytesInCopy, valid };
  }

  testRun(textureCopyView, textureDataLayout, size, { method, dataSize, success, submit = false }) {
    switch (method) {
      case 'WriteTexture': {
        const data = new Uint8Array(dataSize);

        this.expectValidationError(() => {
          this.device.queue.writeTexture(textureCopyView, data, textureDataLayout, size);
        }, !success);

        break;
      }
      case 'CopyBufferToTexture': {
        const buffer = this.device.createBuffer({
          size: dataSize,
          usage: GPUBufferUsage.COPY_SRC,
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToTexture({ buffer, ...textureDataLayout }, textureCopyView, size);

        if (submit) {
          const cmd = encoder.finish();
          this.expectValidationError(() => {
            this.device.queue.submit([cmd]);
          }, !success);
        } else {
          this.expectValidationError(() => {
            encoder.finish();
          }, !success);
        }

        break;
      }
      case 'CopyTextureToBuffer': {
        const buffer = this.device.createBuffer({
          size: dataSize,
          usage: GPUBufferUsage.COPY_DST,
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyTextureToBuffer(textureCopyView, { buffer, ...textureDataLayout }, size);

        if (submit) {
          const cmd = encoder.finish();
          this.expectValidationError(() => {
            this.device.queue.submit([cmd]);
          }, !success);
        } else {
          this.expectValidationError(() => {
            encoder.finish();
          }, !success);
        }

        break;
      }
    }
  }

  // This is a helper function used for creating a texture when we don't have to be very
  // precise about its size as long as it's big enough and properly aligned.
  createAlignedTexture(
    format,
    copySize = { width: 1, height: 1, depth: 1 },
    origin = { x: 0, y: 0, z: 0 }
  ) {
    const info = kSizedTextureFormatInfo[format];
    return this.device.createTexture({
      size: {
        width: Math.max(1, copySize.width + origin.x) * info.blockWidth,
        height: Math.max(1, copySize.height + origin.y) * info.blockHeight,
        depth: Math.max(1, copySize.depth + origin.z),
      },

      format,
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });
  }
}

// For testing divisibility by a number we test all the values returned by this function:
function valuesToTestDivisibilityBy(number) {
  const values = [];
  for (let i = 0; i <= 2 * number; ++i) {
    values.push(i);
  }
  values.push(3 * number);
  return values;
}

// This is a helper function used for expanding test parameters for texel block alignment tests on offset
export function texelBlockAlignmentTestExpanderForOffset({ format }) {
  return poptions(
    'offset',
    valuesToTestDivisibilityBy(kSizedTextureFormatInfo[format].bytesPerBlock)
  );
}

// This is a helper function used for expanding test parameters for texel block alignment tests on rowsPerImage
export function texelBlockAlignmentTestExpanderForRowsPerImage({ format }) {
  return poptions(
    'rowsPerImage',
    valuesToTestDivisibilityBy(kSizedTextureFormatInfo[format].blockHeight)
  );
}

// This is a helper function used for expanding test parameters for texel block alignment tests on origin and size
export function texelBlockAlignmentTestExpanderForValueToCoordinate({ format, coordinateToTest }) {
  switch (coordinateToTest) {
    case 'x':
    case 'width':
      return poptions(
        'valueToCoordinate',
        valuesToTestDivisibilityBy(kSizedTextureFormatInfo[format].blockWidth)
      );

    case 'y':
    case 'height':
      return poptions(
        'valueToCoordinate',
        valuesToTestDivisibilityBy(kSizedTextureFormatInfo[format].blockHeight)
      );

    case 'z':
    case 'depth':
      return poptions('valueToCoordinate', valuesToTestDivisibilityBy(1));
  }
}

// This is a helper function used for filtering test parameters
export function formatCopyableWithMethod({ format, method }) {
  if (method === 'CopyTextureToBuffer') {
    return kSizedTextureFormatInfo[format].copySrc;
  } else {
    return kSizedTextureFormatInfo[format].copyDst;
  }
}
