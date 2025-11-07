import os
import argparse

MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.

def get_bits(value, start, end):
    mask = (1 << (start - end + 1)) - 1
    return (value >> end) & mask

def sign_extend(value, bits):
        if (value >> (bits - 1)) & 1:
            value -= (1 << bits)
        return value

class InsMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        
        with open(ioDir + "/imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]
        """with open(os.path.join(ioDir, "imem.txt")) as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]"""

    def readInstr(self, ReadAddress):
        #read instruction memory
        byte0 = self.IMem[ReadAddress]       # Most significant byte
        byte1 = self.IMem[ReadAddress + 1]
        byte2 = self.IMem[ReadAddress + 2]
        byte3 = self.IMem[ReadAddress + 3]   # Least significant byte

        # Concatenate bytes in big-endian order
        instr_hex = byte0 + byte1 + byte2 + byte3

        instr_word = int(instr_hex, 2)
        #return 32 bit hex val
        return instr_word
        #return instr_hex
        
          
class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + "/dmem.txt") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]

        # Pad DMEM to 1000 bytes (or however large your expected output is)
        while len(self.DMem) < 1000:
            self.DMem.append("00000000")

    def readInstr(self, ReadAddress):
        #read data memory
        byte0 = self.DMem[ReadAddress]       # Most significant byte
        byte1 = self.DMem[ReadAddress + 1]
        byte2 = self.DMem[ReadAddress + 2]
        byte3 = self.DMem[ReadAddress + 3]   # Least significant byte

        #return 32 bit hex val
        # Convert the 32-bit BINARY string to an integer
        data_word = byte0 + byte1 + byte2 + byte3
        data_int = int(data_word, 2)

        # This simulates sign-extension if the loaded number was negative
        # by checking if the MSB (bit 31) is 1.
        if data_int & 0x80000000:
            # Python will handle the negative value correctly if we convert
            # from a 32-bit two's complement representation.
            data_int = data_int - 0x100000000
            
        return data_int
        
    def writeDataMem(self, Address, WriteData):
        # We must handle negative numbers (two's complement)
        if WriteData < 0:
            # Convert negative int to its 32-bit two's complement unsigned
            WriteData = WriteData & 0xFFFFFFFF
            
        # Extract each byte in Big-Endian order
        # Use f-string formatting '08b' to ensure 8 binary digits
        
        # Byte 1 (MSB, [31-24])
        byte1 = f'{(WriteData >> 24) & 0xFF:08b}'
        # Byte 2 ([23-16])
        byte2 = f'{(WriteData >> 16) & 0xFF:08b}'
        # Byte 3 ([15-8])
        byte3 = f'{(WriteData >> 8) & 0xFF:08b}'
        # Byte 4 (LSB, [7-0])
        byte4 = f'{WriteData & 0xFF:08b}'
        
        """print(f"Length of DMEM: {len(self.DMem)}")
        print(f"First 10 entries: {self.DMem[:10]}")
        print(f"Last 10 entries: {self.DMem[-10:]}")"""
        # Write the 8-bit binary strings back into the DMem list
        self.DMem[Address] = byte1
        self.DMem[Address + 1] = byte2
        self.DMem[Address + 2] = byte3
        self.DMem[Address + 3] = byte4
                     
    def outputDataMem(self):
        #resPath = self.ioDir + "\\" + self.id + "_DMEMResult.txt"
        resPath = os.path.join(self.ioDir, f"{self.id}_DMEMResult.txt")
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])

class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        #self.outputFile = os.path.join(ioDir, f"{self.id}_RFResult.txt")
        #self.outputFile = os.path.join(ioDir, "RFResult.txt")
        self.Registers = [0x0 for i in range(32)]

    def readRF(self, Reg_addr): 
        return self.Registers[Reg_addr]
    
    def writeRF(self, Reg_addr, Wrt_reg_data):
        if Reg_addr != 0:
            self.Registers[Reg_addr] = Wrt_reg_data
         
    def outputRF(self, cycle):
        op = ["-"*70+"\n", "State of RF after executing cycle:" + str(cycle) + "\n"]
        #op.extend([str(val)+"\n" for val in self.Registers])
        # This is the corrected part
        for val in self.Registers:
            # Handle negative numbers to get correct 32-bit two's complement
            if val < 0:
                val = (1 << 32) + val
            
            # Format the integer 'val' as a 32-bit binary string,
            # padded with leading zeros ('032b')
            op.extend([f"{val:032b}\n"])
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

class State(object):
    def __init__(self):
        self.IF = {"nop": False, "PC": 0}
        self.ID = {"nop": False, "Instr": 0}
        self.EX = {"nop": False, "Read_data1": 0, "Read_data2": 0, "Imm": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "is_I_type": False, "rd_mem": 0, 
                   "wrt_mem": 0, "alu_op": 0, "wrt_enable": 0}
        self.MEM = {"nop": False, "ALUresult": 0, "Store_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "rd_mem": 0, 
                   "wrt_mem": 0, "wrt_enable": 0}
        self.WB = {"nop": False, "Wrt_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "wrt_enable": 0}

class Core(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem

class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        #super(SingleStageCore, self).__init__(ioDir + "\\SS_", imem, dmem)
        #self.opFilePath = ioDir + "\\StateResult_SS.txt"
        super(SingleStageCore, self).__init__(os.path.join(ioDir, "SS_"), imem, dmem)
        self.opFilePath = os.path.join(ioDir, "StateResult_SS.txt")

    def step(self):
        # Your implementation

        """if self.halted:
            return
        
        # --- NEW HALT LOGIC ---
        # Check if the CURRENT state is 'nop'. If so, we halted last cycle.
        # This is the final cycle (e.g., cycle 6)
        if self.state.IF["nop"]:
            self.halted = True
            # We must print the final "nop" state
            self.myRF.outputRF(self.cycle)
            self.printState(self.state, self.cycle) # Print the current 'nop' state
            
            self.cycle += 1 # Increment cycle one last time
            return
        # --- END NEW LOGIC ---"""
        if self.state.IF["nop"]:
            self.halted = True
            # We must print the final "nop" state
            self.myRF.outputRF(self.cycle)
            self.printState(self.state, self.cycle) # Print the current 'nop' state
            
            self.cycle += 1 # Increment cycle one last time
            return
        
        # --------------------- Fetch  ---------------------
        pc = self.state.IF["PC"]
        instruction = self.ext_imem.readInstr(pc)

        opcode = get_bits(instruction, 6, 0)

        # --------------------- HALT CHECK -------------------
        '''if opcode == 0b1111111:
            self.halted = True
            
            # --- FIX: Update nextState BEFORE printing it ---
            self.nextState.IF["PC"] = pc  # PC stops incrementing
            self.nextState.IF["nop"] = True # Set nop to True
            # --- END FIX ---
            
            # We must still output the state for this final cycle
            self.myRF.outputRF(self.cycle)
            self.printState(self.nextState, self.cycle) # This now prints the correct state
            
            self.state = self.nextState
            self.cycle += 1
            return # Exit the step function'''
        # ----------------- END OF HALT CHECK ----------------
        """ # Check for HALT instruction (Opcode: 1111111)
        if opcode == 0b1111111:
            self.halted = True
            self.myRF.outputRF(self.cycle)
            self.printState(self.nextState, self.cycle)
            self.state = self.nextState
            self.cycle += 1
            return # Exit the step function"""

        # --------------------- DECODE ---------------------
        rd = get_bits(instruction, 11, 7)
        funct3 = get_bits(instruction, 14, 12)
        rs1 = get_bits(instruction, 19, 15)
        rs2 = get_bits(instruction, 24, 20)
        funct7 = get_bits(instruction, 31, 25)

        read_data1 = self.myRF.readRF(rs1)
        read_data2 = self.myRF.readRF(rs2)

        # --- Immediate Generation ---
        imm = 0
        
        # I-Type immediate (for ADDI, ORI, LW, etc.)
        # Bits [31:20]
        if opcode == 0b0010011 or opcode == 0b0000011: 
            imm_i = get_bits(instruction, 31, 20)
            imm = sign_extend(imm_i, 12) # Sign-extend from 12 bits
        elif opcode == 0b0100011:
            imm_s_p1 = get_bits(instruction, 31, 25) # imm[11:5]
            imm_s_p2 = get_bits(instruction, 11, 7)  # imm[4:0]
            imm_s = (imm_s_p1 << 5) | imm_s_p2
            imm = sign_extend(imm_s, 12) # Sign-extend from 12 bits
        
        # --------------------- Execute  ---------------------

        wrt_enable = False    # Does this instruction write to a register?
        dest_reg = rd         # Which register to write to? (Default to rd)
        write_data = 0      # What data to write?
        pc_next = pc + 4  # Default PC increment

        alu_result = 0

        # By default, the next state is not a NOP
        self.nextState.IF["nop"] = False

        # --- FIX: This is the HALT logic ---
        if opcode == 0b1111111:
            self.nextState.IF["PC"] = pc  # PC stops incrementing
            self.nextState.IF["nop"] = True # Set nop for NEXT cycle
            wrt_enable = False


        # R-Type (ADD, SUB, XOR, OR, AND)
        # Opcode: 0110011
        if opcode == 0b0110011:
            wrt_enable = True # R-types always write to a register
            
            if funct3 == 0b000: # ADD or SUB
                if funct7 == 0b0000000: # ADD
                    alu_result = read_data1 + read_data2
                elif funct7 == 0b0100000: # SUB
                    alu_result = read_data1 - read_data2
            
            elif funct3 == 0b100 and funct7 == 0b0000000: # XOR
                alu_result = read_data1 ^ read_data2
            
            elif funct3 == 0b110 and funct7 == 0b0000000: # OR
                alu_result = read_data1 | read_data2
            
            elif funct3 == 0b111 and funct7 == 0b0000000: # AND
                alu_result = read_data1 & read_data2
                
            write_data = alu_result # For R-type, write_data is the ALU result
            self.nextState.IF["PC"] = pc_next

        # I-Type (ADDI, XORI, ORI, ANDI)
        # Opcode: 0010011
        elif opcode == 0b0010011:
            # I-type arithmetic (ADDI, XORI, ORI, ANDI)
            wrt_enable = True # These instructions write to a register
            
            if funct3 == 0b000: # ADDI
                alu_result = read_data1 + imm
            elif funct3 == 0b100: # XORI
                alu_result = read_data1 ^ imm
            elif funct3 == 0b110: # ORI
                alu_result = read_data1 | imm
            elif funct3 == 0b111: # ANDI
                alu_result = read_data1 & imm
                
            write_data = alu_result # For I-type, data to write is ALU result
            self.nextState.IF["PC"] = pc_next

        # I-Type (LW - Load Word)
        # Opcode: 0000011
        elif opcode == 0b0000011:
            # LW (Load Word)
            alu_result = read_data1 + imm
            wrt_enable = True
            write_data = self.ext_dmem.readInstr(alu_result)
            self.nextState.IF["PC"] = pc_next

        # S-Type (SW - Store Word)
        # Opcode: 0100011
        elif opcode == 0b0100011:
            # SW (Store Word)
            alu_result = read_data1 + imm
            wrt_enable = False # SW does not write to a register
            self.ext_dmem.writeDataMem(alu_result, read_data2)
            self.nextState.IF["PC"] = pc_next   

        # B-Type (BEQ, BNE)
        # Opcode: 1100011
        elif opcode == 0b1100011:
            wrt_enable = False # No write-back
            branch_taken = False
            if funct3 == 0b000: # BEQ
                if read_data1 == read_data2:
                    branch_taken = True
            elif funct3 == 0b001: # BNE
                if read_data1 != read_data2:
                    branch_taken = True
            
            if branch_taken:
                pc_next = pc + imm # Overwrite pc_next with branch target
                
            self.nextState.IF["PC"] = pc_next

        # J-Type (JAL)
        # Opcode: 1101111
        elif opcode == 0b1101111:
            wrt_enable = True # JAL writes the return address (PC+4) to rd
            write_data = pc + 4 # The return address
            pc_next = pc + imm # The jump target address
            self.nextState.IF["PC"] = pc_next
        
        else:
            # For unrecognized opcodes, do nothing (NOP)
            self.nextState.IF["PC"] = pc_next

        # --------------------- WRITE BACK ---------------------    
        if wrt_enable:
            self.myRF.writeRF(dest_reg, write_data) # Use 'rd' as destination register

        """if wrt_enable:
            # 1. Mask to 32 bits to handle overflow
            write_data = write_data & 0xFFFFFFFF
            
            # 2. Convert to a signed 32-bit integer if the MSB is 1
            if write_data & 0x80000000:
                write_data = write_data - 0x100000000
                
            self.myRF.writeRF(dest_reg, write_data)"""

        # --------------------- PC Update --------------------
        #self.nextState.IF["PC"] = pc_next

        # --------------------- HALT CHECK -------------------
        """if self.state.IF["nop"]:
            self.halted = True"""

        """
        self.halted = True
        if self.state.IF["nop"]:
            self.halted = True 
        """
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
            
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")
        
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        #super(FiveStageCore, self).__init__(ioDir + "\\FS_", imem, dmem)
        #self.opFilePath = ioDir + "\\StateResult_FS.txt"
        super(FiveStageCore, self).__init__(os.path.join(ioDir, "FS_"), imem, dmem)
        self.opFilePath = os.path.join(ioDir, "StateResult_FS.txt")

    def step(self):
        # Your implementation
        # --------------------- WB stage ---------------------
        
        
        
        # --------------------- MEM stage --------------------
        
        
        
        # --------------------- EX stage ---------------------
        
        
        
        # --------------------- ID stage ---------------------
        
        
        
        # --------------------- IF stage ---------------------
        
        self.halted = True
        if self.state.IF["nop"] and self.state.ID["nop"] and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
            self.halted = True
        
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
        
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.items()])
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.items()])
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.items()])
        printstate.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.items()])
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.items()])

        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

if __name__ == "__main__":
     
    #parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    ioDir = os.path.abspath(args.iodir)
    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)
    
    ssCore = SingleStageCore(ioDir, imem, dmem_ss)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while(True):

        if ssCore.halted and fsCore.halted:
            break

        if not ssCore.halted:
            ssCore.step()
        
        if not fsCore.halted:
            fsCore.step()
    
    # dump SS and FS data mem.
    dmem_ss.outputDataMem()
    dmem_fs.outputDataMem()