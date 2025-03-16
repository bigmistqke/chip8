function createDeferred<T>() {
  let resolve: (value: T) => void = null!
  const promise = new Promise<T>(_resolve => (resolve = _resolve))
  return {
    promise,
    resolve,
  }
}

interface Deferred<T> {
  promise: Promise<T>
  resolve(value: T): void
}

function NOT_IMPLEMENTED_YET() {
  throw 'NOT IMPLEMENTED YET'
}
function INCORRECT_ARG_COUNT(line: string) {
  throw `PARSE ERROR (INCORRECT ARGUMENT COUNT): ${line}`
}
function PARSE_ERROR(line: string) {
  throw `PARSE ERROR: ${line}`
}

function UNKNOWN_OPCODE(opcode: number, mnemonic = 'unknown') {
  const hexOpcode = opcode.toString(16).toUpperCase().padStart(4, '0')
  throw `UNKNOWN OPCODE: 0x${hexOpcode} ${mnemonic}`
}

export const FONT_BASE_ADDRESS = 0x050 // Assuming the font set starts at 0x050 in memory
export const SPRITE_HEIGHT = 5 // Each font sprite has 5 rows
export const WIDTH = 64
export const HEIGHT = 32

const DEFAULT_FONT_SET = [
  // 0
  0b11110000, 0b10010000, 0b10010000, 0b10010000, 0b11110000,
  // 1
  0b00100000, 0b01100000, 0b00100000, 0b00100000, 0b01110000,
  // 2
  0b11110000, 0b00010000, 0b11110000, 0b10000000, 0b11110000,
  // 3
  0b11110000, 0b00010000, 0b11110000, 0b00010000, 0b11110000,
  // 4
  0b10010000, 0b10010000, 0b11110000, 0b00010000, 0b00010000,
  // 5
  0b11110000, 0b10000000, 0b11110000, 0b00010000, 0b11110000,
  // 6
  0b11110000, 0b10000000, 0b11110000, 0b10010000, 0b11110000,
  // 7
  0b11110000, 0b00010000, 0b00100000, 0b01000000, 0b01000000,
  // 8
  0b11110000, 0b10010000, 0b11110000, 0b10010000, 0b11110000,
  // 9
  0b11110000, 0b10010000, 0b11110000, 0b00010000, 0b11110000,
  // A
  0b11110000, 0b10010000, 0b11110000, 0b10010000, 0b10010000,
  // B
  0b11100000, 0b10010000, 0b11100000, 0b10010000, 0b11100000,
  // C
  0b11110000, 0b10000000, 0b10000000, 0b10000000, 0b11110000,
  // D
  0b11100000, 0b10010000, 0b10010000, 0b10010000, 0b11100000,
  // E
  0b11110000, 0b10000000, 0b11110000, 0b10000000, 0b11110000,
  // F
  0b11110000, 0b10000000, 0b11110000, 0b10000000, 0b10000000,
]

export class Chip8WebGLRenderer {
  gl: WebGL2RenderingContext
  texture: WebGLTexture
  constructor(public canvas: HTMLCanvasElement) {
    this.gl = this.canvas.getContext('webgl2')
    if (!this.gl) {
      console.error('WebGL not supported in this browser.')
      return
    }

    this.init()
  }

  init() {
    // Vertex shader
    const vsSource = `
            attribute vec2 aPosition;
            varying highp vec2 vCoords;
            void main() {
                gl_Position = vec4(aPosition, 0.0, 1.0);
                vCoords = aPosition * 0.5 + 0.5;
            }
        `

    // Fragment shader
    const fsSource = `
            precision mediump float;
            varying highp vec2 vCoords;
            uniform sampler2D uTexture;
            void main() {
                gl_FragColor = texture2D(uTexture, vCoords) * vec4(0,1,0,1);
            }
        `

    const shaderProgram = this.createShaderProgram(vsSource, fsSource)
    this.gl.useProgram(shaderProgram)

    // Define vertices for the screen quad
    const vertices = new Float32Array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])

    // Create vertex buffer
    const vertexBuffer = this.gl.createBuffer()
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vertexBuffer)
    this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW)

    // Set up vertex position attribute
    const positionLocation = this.gl.getAttribLocation(shaderProgram, 'aPosition')
    this.gl.enableVertexAttribArray(positionLocation)
    this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0)

    // Create texture
    const texture = this.gl.createTexture()
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture)
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE)
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE)
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST)
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST)
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      64,
      32,
      0,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      null,
    )

    this.texture = texture
  }

  createShaderProgram(vsSource: string, fsSource: string) {
    const gl = this.gl
    const vertexShader = this.loadShader(gl.VERTEX_SHADER, vsSource)
    const fragmentShader = this.loadShader(gl.FRAGMENT_SHADER, fsSource)
    const shaderProgram = gl.createProgram()
    gl.attachShader(shaderProgram, vertexShader)
    gl.attachShader(shaderProgram, fragmentShader)
    gl.linkProgram(shaderProgram)
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
      alert('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram))
      return null
    }
    return shaderProgram
  }

  loadShader(type: number, source: string) {
    const gl = this.gl
    const shader = gl.createShader(type)
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      alert('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader))
      gl.deleteShader(shader)
      return null
    }
    return shader
  }

  updateDisplay(displayBuffer: Uint8Array) {
    // Convert CHIP-8 display buffer (1 bit per pixel) to a WebGL texture
    const pixelData = new Uint8Array(64 * 32 * 4) // RGBA format
    for (let i = 0; i < displayBuffer.length; i++) {
      const val = displayBuffer[i] ? 255 : 0
      pixelData[i * 4] = val // R
      pixelData[i * 4 + 1] = val // G
      pixelData[i * 4 + 2] = val // B
      pixelData[i * 4 + 3] = 255 // A
    }

    const gl = this.gl
    gl.bindTexture(gl.TEXTURE_2D, this.texture)
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 64, 32, gl.RGBA, gl.UNSIGNED_BYTE, pixelData)
  }

  render() {
    const gl = this.gl
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }
}

export class Chip8Audio {
  runtime: Chip8
  audioContext: AudioContext
  oscillator: OscillatorNode
  gainNode: GainNode

  constructor() {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    this.oscillator = null
    this.gainNode = this.audioContext.createGain()
    this.gainNode.connect(this.audioContext.destination)
    this.gainNode.gain.value = 0 // Start with the sound off
  }

  start(frequency = 440) {
    if (this.oscillator) {
      this.stop() // Stop the current tone before starting a new one
    }
    this.oscillator = this.audioContext.createOscillator()
    this.oscillator.type = 'square' // Square wave for a classic beep sound
    this.oscillator.frequency.value = frequency // Frequency in hertz
    this.oscillator.connect(this.gainNode)
    this.oscillator.start()
    this.gainNode.gain.value = 0.5 // Set the volume level
  }

  stop() {
    if (this.oscillator) {
      this.oscillator.stop()
      this.oscillator.disconnect()
      this.oscillator = null
    }
    this.gainNode.gain.value = 0 // Mute the sound
  }
}

export class Chip8 {
  memory = new Uint8Array(4096) // 4KB RAM
  registers = new Uint8Array(16) // V0 - VF
  I = 0 // Index register
  PC = 0x200 // Programs start at 0x200
  stack = new Uint16Array(16)
  SP = 0 // Stack pointer
  display = new Uint8Array(WIDTH * HEIGHT) // 64x32 monochrome pixels
  keypad = new Uint8Array(16) // 16-key input
  delayTimer = 0
  soundTimer = 0
  awaitKeyPress: Deferred<number> | undefined

  constructor(
    public renderer: Chip8WebGLRenderer,
    public audio: Chip8Audio,
  ) {
    this.memory.set(DEFAULT_FONT_SET, 0x050)
  }

  loadProgram(program: Uint8Array) {
    this.memory.set(program, 0x200) // Load program into memory at 0x200
  }

  pressKey(key: number) {
    this.awaitKeyPress?.resolve(key)
  }

  async cycle() {
    let opcode = (this.memory[this.PC] << 8) | this.memory[this.PC + 1]
    this.PC += 2
    await this.executeOpcode(opcode)
    this.renderer.updateDisplay(this.display)
    this.renderer.render()
  }

  async executeOpcode(opcode: number) {
    let firstNibble = opcode & 0xf000
    let lastNibble = opcode & 0x000f
    switch (firstNibble) {
      case 0x0000: {
        switch (opcode) {
          // CLS - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#00E0
          case 0x00e0: {
            // Clear the display.
            this.display.fill(0)
            break
          }
          // RET - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#00EE
          case 0x00ee: {
            // Return from a subroutine.
            // The interpreter sets the program counter to the address at the top of the stack, then subtracts 1 from the stack pointer.
            this.SP--
            this.PC = this.stack[this.SP]
            break
          }
          default: {
            UNKNOWN_OPCODE(opcode)
          }
        }
      }
      // JP addr - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#1nnn
      case 0x1000: {
        // Jump to location nnn.
        // The interpreter sets the program counter to nnn.
        const address = opcode & 0x0fff
        this.PC = address
        break
      }
      // CALL addr - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#2nnn
      case 0x2000: {
        // Call subroutine at nnn.
        // The interpreter increments the stack pointer, then puts the current PC on the top of the stack. The PC is then set to nnn.
        this.stack[this.SP] = this.PC
        this.SP++
        this.PC = opcode & 0x0fff
        break
      }
      // SE Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#3xkk
      case 0x3000: {
        // Skip next instruction if Vx = kk.
        // The interpreter compares register Vx to kk, and if they are equal, increments the program counter by 2.
        const x = (opcode & 0x0f00) >> 8
        const kk = opcode & 0x00ff
        if (this.registers[x] === kk) {
          this.PC += 2
        }
        break
      }
      // SNE Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#4xkk
      case 0x4000: {
        // Skip next instruction if Vx != kk.
        // The interpreter compares register Vx to kk, and if they are not equal, increments the program counter by 2.
        const x = (opcode & 0x0f00) >> 8
        const kk = opcode & 0x00ff
        if (this.registers[x] !== kk) {
          this.PC += 2
        }
        break
      }
      // SE Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#5xy0
      case 0x5000: {
        if (lastNibble !== 0) {
          UNKNOWN_OPCODE(opcode, 'SE Vx, Vy')
        }
        // Skip next instruction if Vx = Vy.
        // The interpreter compares register Vx to register Vy, and if they are equal, increments the program counter by 2.
        const x = (opcode & 0x0f00) >> 8
        const y = (opcode & 0x00f0) >> 4
        if (this.registers[x] === this.registers[y]) {
          this.PC += 2
        }
        break
      }
      // LD Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#6xkk
      case 0x6000: {
        // Set Vx = kk.
        // The interpreter puts the value kk into register Vx.
        const x = (opcode & 0x0f00) >> 8
        const kk = opcode & 0x00ff
        this.registers[x] = kk
        break
      }
      // ADD Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#7xkk
      case 0x7000: {
        // Set Vx = Vx + kk.
        // Adds the value kk to the value of register Vx, then stores the result in Vx.
        const x = (opcode & 0x0f00) >> 8
        const kk = opcode & 0x00ff
        this.registers[x] = (this.registers[x] + kk) & 0xff
        break
      }
      case 0x8000: {
        const x = (opcode & 0x0f00) >> 8
        const y = (opcode & 0x00f0) >> 4
        switch (lastNibble) {
          // LD Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy0
          case 0: {
            // Set Vx = Vy.
            // Stores the value of register Vy in register Vx.
            this.registers[x] = this.registers[y]
            break
          }
          // OR Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy1
          case 1: {
            // Set Vx = Vx OR Vy.
            // Performs a bitwise OR on the values of Vx and Vy, then stores the result in Vx. A bitwise OR compares the corrseponding bits from two values, and if either bit is 1, then the same bit in the result is also 1. Otherwise, it is 0.
            this.registers[x] |= this.registers[y]
            break
          }
          // AND Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy2
          case 2: {
            // Set Vx = Vx AND Vy.
            // Performs a bitwise AND on the values of Vx and Vy, then stores the result in Vx. A bitwise AND compares the corrseponding bits from two values, and if both bits are 1, then the same bit in the result is also 1. Otherwise, it is 0.
            this.registers[x] &= this.registers[y]
            break
          }
          // XOR Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy3
          case 3: {
            // Set Vx = Vx XOR Vy.
            // Performs a bitwise exclusive OR on the values of Vx and Vy, then stores the result in Vx. An exclusive OR compares the corrseponding bits from two values, and if the bits are not both the same, then the corresponding bit in the result is set to 1. Otherwise, it is 0.
            this.registers[x] ^= this.registers[y]
            break
          }
          // ADD Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy4
          case 4: {
            // Set Vx = Vx + Vy, set VF = carry.
            // Performs a bitwise exclusive OR on the values of Vx and Vy, then stores the result in Vx. An exclusive OR compares the corrseponding bits from two values, and if the bits are not both the same, then the corresponding bit in the result is set to 1. Otherwise, it is 0.
            const value = this.registers[x] + this.registers[y]
            this.registers[0xf] = value > 255 ? 1 : 0
            this.registers[x] = value & 0xff
            break
          }
          // SUB Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy5
          case 5: {
            // Set Vx = Vx - Vy, set VF = NOT borrow.
            // If Vx > Vy, then VF is set to 1, otherwise 0. Then Vy is subtracted from Vx, and the results stored in Vx.
            this.registers[0xf] = this.registers[x] > this.registers[y] ? 1 : 0
            this.registers[x] = (this.registers[x] - this.registers[y]) & 0xff
            break
          }
          // SHR Vx {, Vy} - - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy6
          case 6: {
            // Set Vx = Vx SHR 1.
            // If the least-significant bit of Vx is 1, then VF is set to 1, otherwise 0. Then Vx is divided by 2.
            const lsb = this.registers[x] & 0x01
            this.registers[0xf] = lsb
            this.registers[x] >>= 1
            break
          }
          // SUBN Vx, Vy - - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy7
          case 7: {
            // Set Vx = Vy - Vx, set VF = NOT borrow.
            // If Vy > Vx, then VF is set to 1, otherwise 0. Then Vx is subtracted from Vy, and the results stored in Vx.
            this.registers[0xf] = this.registers[x] > this.registers[y] ? 1 : 0
            this.registers[x] = (this.registers[y] - this.registers[x]) & 0xff
            break
          }
          // SHL Vx {, Vy} - - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy8
          case 8: {
            // Set Vx = Vx SHL 1.
            // If the most-significant bit of Vx is 1, then VF is set to 1, otherwise to 0. Then Vx is multiplied by 2.
            const msb = (this.registers[x] & 0x80) >> 7
            this.registers[0xf] = msb
            this.registers[x] <<= 1
            break
          }
          // SNE Vx, Vy - - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy9
          case 9: {
            if (this.registers[x] !== this.registers[y]) {
              this.PC += 2
            }
            break
          }
        }
        break
      }
      // LD I, addr - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Annn
      case 0xa000: {
        // Set I = nnn.
        // The value of register I is set to nnn.
        const nnn = opcode & 0x0fff
        this.I = nnn
        break
      }
      // JP V0, addr - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Bnnn
      case 0xb000: {
        // Jump to location nnn + V0.
        // The program counter is set to nnn plus the value of V0.
        const nnn = opcode & 0x0fff
        this.PC = this.registers[0] + nnn
        break
      }
      // RND Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Cxkk
      case 0xc000: {
        // Set Vx = random byte AND kk.
        // The interpreter generates a random number from 0 to 255, which is then ANDed with the value kk. The results are stored in Vx. See instruction 8xy2 for more information on AND.
        const x = (opcode & 0x0f00) >> 8
        const kk = opcode & 0x00ff
        this.registers[x] = Math.floor(Math.random() * 256) & kk
        break
      }
      // DRW Vx, Vy, nibble - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Dxyn
      case 0xd000: {
        // Display n-byte sprite starting at memory location I at (Vx, Vy), set VF = collision.
        //
        // The interpreter reads n bytes from memory, starting at the address stored in I.
        // These bytes are then displayed as sprites on screen at coordinates (Vx, Vy).
        // Sprites are XORed onto the existing screen.
        // If this causes any pixels to be erased, VF is set to 1, otherwise it is set to 0.
        // If the sprite is positioned so part of it is outside the coordinates of the display, it wraps around to the opposite side of the screen.
        // See instruction 8xy3 for more information on XOR, and section 2.4, Display, for more information on the Chip-8 screen and sprites.
        const x = this.registers[(opcode & 0x0f00) >> 8]
        const y = this.registers[(opcode & 0x00f0) >> 4]
        const n = opcode & 0x000f

        // Reset VF to 0 (no collision)
        this.registers[0xf] = 0

        for (let i = 0; i < n; i++) {
          const spriteRow = this.memory[this.I + i] // Read byte (sprite row) from memory
          for (let j = 0; j < 8; j++) {
            const bitMask = 0x80 >> j
            const bitOffset = 7 - j
            const pixel = (spriteRow & bitMask) >> bitOffset

            const _x = (x + j) % WIDTH
            const _y = ((y + i) % HEIGHT) * WIDTH
            const idx = _x + _y

            if (pixel) {
              // Check for collision
              if (this.display[idx] === 1) {
                this.registers[0xf] = 1
              }
              this.display[idx] ^= 1 // XOR the pixel with the display buffer
            }
          }
        }
        break
      }
      case 0xe000: {
        const x = (opcode & 0x0f00) >> 8
        const nn = opcode & 0x00ff
        switch (nn) {
          // SKP Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Ex9E
          case 0x9e: {
            // Skip next instruction if key with the value of Vx is pressed.
            // Checks the keyboard, and if the key corresponding to the value of Vx is currently in the down position, PC is increased by 2.
            if (this.keypad[this.registers[x]]) {
              this.PC += 2
            }
            break
          }
          // SKNP Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#ExA1
          case 0xa1: {
            // Skip next instruction if key with the value of Vx is not pressed.
            // Checks the keyboard, and if the key corresponding to the value of Vx is currently in the up position, PC is increased by 2.
            if (!this.keypad[this.registers[x]]) {
              this.PC += 2
            }
            break
          }
          default:
            UNKNOWN_OPCODE(opcode)
        }
      }
      case 0xf000: {
        const x = (opcode & 0x0f00) >> 8
        const nn = opcode & 0x00ff
        switch (nn) {
          // LD Vx, DT - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx07
          case 0x07: {
            // Set Vx = delay timer value.
            // The value of DT is placed into Vx.
            this.registers[x] = this.delayTimer
            break
          }
          // LD Vx, K - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx0A
          case 0x0a: {
            // Wait for a key press, store the value of the key in Vx.
            // All execution stops until a key is pressed, then the value of that key is stored in Vx.
            this.awaitKeyPress = createDeferred<number>()
            this.registers[x] = await this.awaitKeyPress.promise
            this.awaitKeyPress = undefined
            break
          }
          // LD DT, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx15
          case 0x15: {
            // Set delay timer = Vx.
            // DT is set equal to the value of Vx.
            this.delayTimer = this.registers[x]
            break
          }
          // LD ST, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx18
          case 0x18: {
            // Set sound timer = Vx.
            // ST is set equal to the value of Vx.
            this.soundTimer = this.registers[x]
            break
          }
          // ADD I, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx1E
          case 0x1e: {
            // Set I = I + Vx.
            // The values of I and Vx are added, and the results are stored in I.
            this.I = (this.I + this.registers[x]) & 0xfff
            break
          }
          // LD F, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx29
          case 0x29: {
            // Set I = location of sprite for digit Vx.
            // The value of I is set to the location for the hexadecimal sprite corresponding to the value of Vx.
            // See section 2.4, Display, for more information on the Chip-8 hexadecimal font.
            this.I = FONT_BASE_ADDRESS + this.registers[x] * SPRITE_HEIGHT
            break
          }
          // LD B, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx33
          case 0x33: {
            // Store BCD representation of Vx in memory locations I, I+1, and I+2.
            // The interpreter takes the decimal value of Vx, and places the hundreds digit in memory at location in I, the tens digit at location I+1, and the ones digit at location I+2.
            this.memory[this.I] = Math.floor(this.registers[x] / 100)
            this.memory[this.I + 1] = Math.floor((this.registers[x] % 100) / 10)
            this.memory[this.I + 2] = this.registers[x] % 10
            break
          }
          // LD [I], Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx55
          case 0x55: {
            // Store registers V0 through Vx in memory starting at location I.
            // The interpreter copies the values of registers V0 through Vx into memory, starting at the address in I.
            for (let i = 0; i <= x; i++) {
              this.memory[this.I + i] = this.registers[i]
            }
            break
          }
          // LD Vx, [I] - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx65
          case 0x65: {
            // Read registers V0 through Vx from memory starting at location I.
            // The interpreter reads values from memory starting at location I into registers V0 through Vx.
            for (let i = 0; i <= x; i++) {
              this.registers[i] = this.memory[this.I + i]
            }
            break
          }
          default:
            UNKNOWN_OPCODE(opcode)
        }
      }
    }
  }
}

function stripComment(line: string) {
  const commentIndex = line.indexOf(';') // Find the start of a comment
  return commentIndex !== -1 ? line.substring(0, commentIndex).trim() : line.trim() // Remove comment
}

function deserializeNumber(value: string) {
  if (value.startsWith('0x')) {
    return parseInt(value, 16)
  }
  return parseInt(value, 10)
}

function isRegister(value: string): value is `V${string}` {
  return value.startsWith('V')
}

function getRegister(value: `V${string}`) {
  return parseInt(value.substring(1), 16)
}

export function assemble(chip8Assembly: string) {
  const lines = chip8Assembly.split('\n')
  const binaryOutput = []

  // Second pass to translate assembly to binary
  lines.forEach(line => {
    line = stripComment(line)
    let [instruction, ...tokens] = line.split(/\s+/)
    tokens = tokens.map(token => token.replace(/,$/, ''))

    if (instruction && !instruction.endsWith(':')) {
      // Ignore labels and empty lines
      switch (instruction.toUpperCase()) {
        case 'BD': {
          // CUSTOM TOKEN
          if (tokens.length !== 1) {
            return INCORRECT_ARG_COUNT(line)
          }
          binaryOutput.push(tokens[0])
          return
        }
        case 'RET': {
          if (tokens.length !== 0) {
            return INCORRECT_ARG_COUNT(line)
          }
          // RET - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#00EE
          binaryOutput.push(0x00, 0xee)
          break
        }
        case 'CLS': {
          if (tokens.length !== 0) {
            return INCORRECT_ARG_COUNT(line)
          }
          // CLS - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#00E0
          binaryOutput.push(0x00, 0xe0)
          break
        }
        case 'CALL': {
          if (tokens.length !== 1) {
            return INCORRECT_ARG_COUNT(line)
          }
          const byte = deserializeNumber(tokens[0])
          if (!isNaN(byte)) {
            // CALL addr - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#2nnn
            binaryOutput.push(0x20 | (byte >> 8), byte & 0xff)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'SE': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0]) && isRegister(tokens[1])) {
            const x = getRegister(tokens[0])
            const y = getRegister(tokens[1])
            // SE Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#5xy0
            binaryOutput.push(0x50 | x, y << 4)
            return
          }
          const byte = deserializeNumber(tokens[1])
          if (isRegister(tokens[0]) && !isNaN(byte)) {
            const x = getRegister(tokens[0])
            // SE Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#3xkk
            binaryOutput.push(0x30 | x, byte)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'SNE': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          const byte = deserializeNumber(tokens[1])
          if (isRegister(tokens[0]) && !isNaN(byte)) {
            const x = getRegister(tokens[0])
            // SNE Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#4xkk
            binaryOutput.push(0x40 | x, byte)
          }
          return PARSE_ERROR(line)
        }

        case 'OR': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0]) && isRegister(tokens[1])) {
            const x = getRegister(tokens[0])
            const y = getRegister(tokens[1])
            // OR Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy1
            binaryOutput.push(0x80 | x, (y << 8) | 1)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'AND': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0]) && isRegister(tokens[1])) {
            const x = getRegister(tokens[0])
            const y = getRegister(tokens[1])
            // AND Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy2
            binaryOutput.push(0x80 | x, (y << 8) | 2)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'XOR': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0]) && isRegister(tokens[1])) {
            const x = getRegister(tokens[0])
            const y = getRegister(tokens[1])
            // XOR Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy3
            binaryOutput.push(0x80 | x, (y << 8) | 3)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'ADD': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (tokens[0] === 'I' && isRegister(tokens[1])) {
            // ADD I, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx1E
            const x = getRegister(tokens[1])
            binaryOutput.push(0xf0 | x, 0x1e)
            return
          }
          if (isRegister(tokens[0]) && isRegister(tokens[1])) {
            const x = getRegister(tokens[0])
            const y = getRegister(tokens[1])
            // ADD Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy4
            binaryOutput.push(0x80 | x, (y << 4) | 4)
            return
          }
          const byte = deserializeNumber(tokens[1])
          if (isRegister(tokens[0]) && !isNaN(byte)) {
            const x = getRegister(tokens[0])
            // ADD Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#7xkk
            binaryOutput.push(0x70 | x, byte)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'SUB': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0]) && isRegister(tokens[1])) {
            const x = getRegister(tokens[0])
            const y = getRegister(tokens[1])
            // SUB Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy5
            binaryOutput.push(0x80 | x, (y << 4) | 5)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'SHR': {
          if (tokens.length !== 1) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0])) {
            const x = getRegister(tokens[0])
            // SHR Vx {, Vy} - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy6
            // NOTE: 'modern' implementations ignore Vy
            binaryOutput.push(0x80 | x, (x << 4) | 6)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'SUBN': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0]) && isRegister(tokens[1])) {
            const x = getRegister(tokens[0])
            const y = getRegister(tokens[1])
            // SUBN Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy7
            binaryOutput.push(0x80 | x, (y << 4) | 7)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'SHL': {
          if (tokens.length !== 1) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0])) {
            const x = getRegister(tokens[0])
            // SHL Vx {, Vy} - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xyE
            // NOTE: 'modern' implementations ignore Vy
            binaryOutput.push(0x80 | x, (x << 4) | 0xe)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'RND': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          const byte = deserializeNumber(tokens[1])
          if (isRegister(tokens[0]) && !isNaN(byte)) {
            // RND Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Cxkk
            const x = getRegister(tokens[0])
            binaryOutput.push(0xc0 | x, byte)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'DRW': {
          if (tokens.length !== 3) {
            return INCORRECT_ARG_COUNT(line)
          }
          const nibble = parseInt(tokens[2], 16)
          if (isRegister(tokens[0]) && isRegister(tokens[1]) && !isNaN(nibble)) {
            const x = getRegister(tokens[0])
            const y = getRegister(tokens[1])
            // DRW Vx, Vy, nibble - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Dxyn
            binaryOutput.push(0xd0 | x, (y << 4) | nibble)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'SKP': {
          if (tokens.length !== 1) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0])) {
            // SKP Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Ex9E
            const x = getRegister(tokens[0])
            binaryOutput.push(0xe0 | x, 0x9e)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'SKNP': {
          if (tokens.length !== 1) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (isRegister(tokens[0])) {
            // SKNP Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#ExA1
            const x = getRegister(tokens[0])
            binaryOutput.push(0xe0 | x, 0xa1)
            return
          }
          return PARSE_ERROR(line)
        }
        case 'JP': {
          if (tokens.length !== 1 && tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          if (tokens.length === 1) {
            const byte = deserializeNumber(tokens[0])
            if (!isNaN(byte)) {
              // JP addr - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#1nnn
              binaryOutput.push(0x10 | (byte >> 8), byte & 0xff)
              return
            }
          } else if (tokens.length === 2) {
            const [x, addr] = tokens
            const byte = deserializeNumber(addr)
            if (x === 'V0' && !isNaN(byte)) {
              // JP V0, addr - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Bnnn
              binaryOutput.push(0xb0 | (byte >> 8), byte & 0xff)
              return
            }
          }
          return PARSE_ERROR(line)
        }

        case 'LD': {
          if (tokens.length !== 2) {
            return INCORRECT_ARG_COUNT(line)
          }
          let [target, value] = tokens
          switch (target) {
            case 'F': {
              if (isRegister(value)) {
                // LD F, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx29
                const x = getRegister(value)
                binaryOutput.push(0xf0 | x, 0x29)
                return
              }
              return PARSE_ERROR(line)
            }
            case 'B': {
              if (isRegister(value)) {
                // LD B, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx33
                const x = getRegister(value)
                binaryOutput.push(0xf0 | x, 0x33)
                return
              }
              return PARSE_ERROR(line)
            }
            case '[I]': {
              if (isRegister(value)) {
                // LD [I], Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx55
                const x = getRegister(value)
                binaryOutput.push(0xf0 | x, 0x55)
                return
              }
              return PARSE_ERROR(line)
            }
            case 'I': {
              const byte = deserializeNumber(value)
              if (!isNaN(byte)) {
                // LD I, addr - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Annn
                binaryOutput.push(0xa0 | (byte >> 8), byte & 0xff)
                return
              }
              return PARSE_ERROR(line)
            }
            case 'DT': {
              if (isRegister(value)) {
                // LD DT, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx15
                const x = getRegister(value)
                binaryOutput.push(0xf0 | x, 0x15)
                return
              }
              return PARSE_ERROR(line)
            }
            case 'ST': {
              if (isRegister(value)) {
                // LD ST, Vx - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx18
                const x = getRegister(value)
                binaryOutput.push(0xf0 | x, 0x18)
                return
              }
              return PARSE_ERROR(line)
            }
            default:
              if (!isRegister(target)) {
                throw `Unknown target ${target} for instruction ${instruction}`
              }
              const x = getRegister(target)
              switch (value) {
                case 'DT': {
                  // LD Vx, DT - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx07
                  binaryOutput.push(0xf0 | x, 0x07)
                  break
                }
                case 'K': {
                  // LD Vx, K - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx0A
                  binaryOutput.push(0xf0 | x, 0x0a)
                  break
                }
                case '[I]': {
                  // LD Vx, [I] - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx65
                  binaryOutput.push(0xf0 | x, 0x65)
                  break
                }
                case 'R': {
                  // LD Vx, R - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#Fx85
                  binaryOutput.push(0xf0 | x, 0x85)
                  break
                }
                default: {
                  if (isRegister(value)) {
                    // LD Vx, Vy - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#8xy0
                    // 8xy0
                    const y = getRegister(value)
                    binaryOutput.push(0x80 | x, y << 4)
                    return
                  }
                  const byte = deserializeNumber(value)
                  console.log('byte', byte)
                  if (!isNaN(byte)) {
                    // LD Vx, byte - http://devernay.free.fr/hacks/chip8/C8TECH10.HTM#6xkk
                    binaryOutput.push(0x60 | x, byte)
                    return
                  }
                  return PARSE_ERROR(line)
                }
              }
              break
          }
          break
        }
      }
    }
  })

  return new Uint8Array(binaryOutput)
}
