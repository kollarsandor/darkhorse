module top_level (
    input wire clk,
    input wire rst_n,
    input  wire [15:0] axi_awaddr,
    input  wire        axi_awvalid,
    output wire        axi_awready,
    input  wire [2:0]  axi_awprot,
    input  wire [31:0] axi_wdata,
    input  wire [3:0]  axi_wstrb,
    input  wire        axi_wvalid,
    output wire        axi_wready,
    output wire [1:0]  axi_bresp,
    output wire        axi_bvalid,
    input  wire        axi_bready,
    input  wire [15:0] axi_araddr,
    input  wire        axi_arvalid,
    output wire        axi_arready,
    input  wire [2:0]  axi_arprot,
    output wire [31:0] axi_rdata,
    output wire [1:0]  axi_rresp,
    output wire        axi_rvalid,
    input  wire        axi_rready,
    output wire [31:0] mem_addr,
    output wire [15:0] mem_wdata,
    input  wire [15:0] mem_rdata,
    output wire        mem_we,
    output wire        mem_oe,
    output wire        mem_ce,
    input  wire        mem_ready,
    output wire [7:0]  led_status,
    output wire        led_error,
    output wire        irq_out
);
    wire reset;
    reg rst_sync1, rst_sync2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_sync1 <= 1'b0;
        else rst_sync1 <= 1'b1;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_sync2 <= 1'b0;
        else rst_sync2 <= rst_sync1;
    end
    assign reset = !rst_sync2;
    reg [31:0] control_reg;
    reg [31:0] status_reg;
    reg [31:0] config_reg;
    reg [31:0] result_reg;
    wire [63:0] ssi_search_key;
    wire [31:0] ssi_root_addr;
    wire        ssi_start;
    wire [31:0] ssi_result_addr;
    wire        ssi_found;
    wire [7:0]  ssi_depth;
    wire        ssi_done;
    wire [31:0] ssi_mem_addr;
    wire        ssi_mem_req;
    wire        ssi_node_valid;
    wire [63:0] ranker_query_hash;
    wire [63:0] ranker_segment_id;
    wire [63:0] ranker_segment_pos;
    wire [31:0] ranker_base_score;
    wire        ranker_valid;
    wire [31:0] ranker_final_score;
    wire [15:0] ranker_rank;
    wire        ranker_done;
    wire [31:0] arbiter_mem_addr;
    wire [15:0] arbiter_mem_wdata;
    wire        arbiter_mem_we;
    wire        arbiter_mem_req;
    wire        arbiter_grant;
    wire [3:0]  client_req;
    wire [3:0]  client_grant;
    localparam ADDR_CONTROL   = 16'h0000;
    localparam ADDR_STATUS    = 16'h0004;
    localparam ADDR_CONFIG    = 16'h0008;
    localparam ADDR_RESULT    = 16'h000C;
    localparam ADDR_SSI_KEY_L = 16'h0010;
    localparam ADDR_SSI_KEY_H = 16'h0014;
    localparam ADDR_SSI_ROOT  = 16'h0018;
    localparam ADDR_SSI_RES   = 16'h001C;
    localparam ADDR_RNK_HASH_L= 16'h0020;
    localparam ADDR_RNK_HASH_H= 16'h0024;
    localparam ADDR_RNK_SEG_L = 16'h0028;
    localparam ADDR_RNK_SEG_H = 16'h002C;
    localparam ADDR_RNK_POS_L = 16'h0030;
    localparam ADDR_RNK_POS_H = 16'h0034;
    localparam ADDR_RNK_SCORE = 16'h0038;
    localparam ADDR_RNK_RES   = 16'h003C;
    reg aw_ready_reg;
    reg w_ready_reg;
    reg b_valid_reg;
    reg [1:0] b_resp_reg;
    reg ar_ready_reg;
    reg r_valid_reg;
    reg [1:0] r_resp_reg;
    reg [15:0] aw_addr_reg;
    reg aw_valid_reg;
    reg [15:0] ar_addr_reg;
    reg [31:0] r_data_reg;
    assign axi_awready = aw_ready_reg;
    assign axi_wready = w_ready_reg;
    assign axi_bvalid = b_valid_reg;
    assign axi_bresp = b_resp_reg;
    assign axi_arready = ar_ready_reg;
    assign axi_rvalid = r_valid_reg;
    assign axi_rresp = r_resp_reg;
    assign axi_rdata = r_data_reg;
    reg [63:0] ssi_key_reg;
    reg [31:0] ssi_root_reg;
    reg [31:0] ssi_result_reg;
    reg [63:0] ranker_hash_reg;
    reg [63:0] ranker_seg_reg;
    reg [63:0] ranker_pos_reg;
    reg [31:0] ranker_score_reg;
    reg [31:0] ranker_result_reg;
    wire aw_handshake = axi_awvalid && axi_awready;
    wire w_handshake = axi_wvalid && axi_wready;
    wire b_handshake = axi_bvalid && axi_bready;
    wire ar_handshake = axi_arvalid && axi_arready;
    wire r_handshake = axi_rvalid && axi_rready;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            aw_ready_reg <= 1'b1;
            w_ready_reg <= 1'b1;
            b_valid_reg <= 1'b0;
            b_resp_reg <= 2'b00;
            aw_addr_reg <= 16'h0;
            aw_valid_reg <= 1'b0;
            control_reg <= 32'h0;
            config_reg <= 32'h0;
            ssi_key_reg <= 64'h0;
            ssi_root_reg <= 32'h0;
            ranker_hash_reg <= 64'h0;
            ranker_seg_reg <= 64'h0;
            ranker_pos_reg <= 64'h0;
            ranker_score_reg <= 32'h0;
        end else begin
            if (aw_handshake && !aw_valid_reg) begin
                aw_addr_reg <= axi_awaddr;
                aw_valid_reg <= 1'b1;
                aw_ready_reg <= 1'b0;
            end
            if (w_handshake && aw_valid_reg && !b_valid_reg) begin
                b_valid_reg <= 1'b1;
                aw_valid_reg <= 1'b0;
                w_ready_reg <= 1'b0;
                b_resp_reg <= 2'b00;
                case (aw_addr_reg)
                    ADDR_CONTROL: begin
                        if (axi_wstrb[0]) control_reg[7:0] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) control_reg[15:8] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) control_reg[23:16] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) control_reg[31:24] <= axi_wdata[31:24];
                    end
                    ADDR_CONFIG: begin
                        if (axi_wstrb[0]) config_reg[7:0] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) config_reg[15:8] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) config_reg[23:16] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) config_reg[31:24] <= axi_wdata[31:24];
                    end
                    ADDR_SSI_KEY_L: begin
                        if (axi_wstrb[0]) ssi_key_reg[7:0] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ssi_key_reg[15:8] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ssi_key_reg[23:16] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ssi_key_reg[31:24] <= axi_wdata[31:24];
                    end
                    ADDR_SSI_KEY_H: begin
                        if (axi_wstrb[0]) ssi_key_reg[39:32] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ssi_key_reg[47:40] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ssi_key_reg[55:48] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ssi_key_reg[63:56] <= axi_wdata[31:24];
                    end
                    ADDR_SSI_ROOT: begin
                        if (axi_wstrb[0]) ssi_root_reg[7:0] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ssi_root_reg[15:8] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ssi_root_reg[23:16] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ssi_root_reg[31:24] <= axi_wdata[31:24];
                    end
                    ADDR_RNK_HASH_L: begin
                        if (axi_wstrb[0]) ranker_hash_reg[7:0] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ranker_hash_reg[15:8] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ranker_hash_reg[23:16] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ranker_hash_reg[31:24] <= axi_wdata[31:24];
                    end
                    ADDR_RNK_HASH_H: begin
                        if (axi_wstrb[0]) ranker_hash_reg[39:32] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ranker_hash_reg[47:40] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ranker_hash_reg[55:48] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ranker_hash_reg[63:56] <= axi_wdata[31:24];
                    end
                    ADDR_RNK_SEG_L: begin
                        if (axi_wstrb[0]) ranker_seg_reg[7:0] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ranker_seg_reg[15:8] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ranker_seg_reg[23:16] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ranker_seg_reg[31:24] <= axi_wdata[31:24];
                    end
                    ADDR_RNK_SEG_H: begin
                        if (axi_wstrb[0]) ranker_seg_reg[39:32] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ranker_seg_reg[47:40] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ranker_seg_reg[55:48] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ranker_seg_reg[63:56] <= axi_wdata[31:24];
                    end
                    ADDR_RNK_POS_L: begin
                        if (axi_wstrb[0]) ranker_pos_reg[7:0] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ranker_pos_reg[15:8] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ranker_pos_reg[23:16] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ranker_pos_reg[31:24] <= axi_wdata[31:24];
                    end
                    ADDR_RNK_POS_H: begin
                        if (axi_wstrb[0]) ranker_pos_reg[39:32] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ranker_pos_reg[47:40] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ranker_pos_reg[55:48] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ranker_pos_reg[63:56] <= axi_wdata[31:24];
                    end
                    ADDR_RNK_SCORE: begin
                        if (axi_wstrb[0]) ranker_score_reg[7:0] <= axi_wdata[7:0];
                        if (axi_wstrb[1]) ranker_score_reg[15:8] <= axi_wdata[15:8];
                        if (axi_wstrb[2]) ranker_score_reg[23:16] <= axi_wdata[23:16];
                        if (axi_wstrb[3]) ranker_score_reg[31:24] <= axi_wdata[31:24];
                    end
                    default: b_resp_reg <= 2'b10;
                endcase
            end
            if (b_handshake) begin
                b_valid_reg <= 1'b0;
            end
            if (!aw_valid_reg && !b_valid_reg) begin
                aw_ready_reg <= 1'b1;
                w_ready_reg <= 1'b1;
            end
        end
    end
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            ar_ready_reg <= 1'b1;
            r_valid_reg <= 1'b0;
            r_resp_reg <= 2'b00;
            ar_addr_reg <= 16'h0;
            r_data_reg <= 32'h0;
        end else begin
            if (ar_handshake && !r_valid_reg) begin
                ar_addr_reg <= axi_araddr;
                r_valid_reg <= 1'b1;
                ar_ready_reg <= 1'b0;
                r_resp_reg <= 2'b00;
                case (axi_araddr)
                    ADDR_CONTROL: r_data_reg <= control_reg;
                    ADDR_STATUS: r_data_reg <= status_reg;
                    ADDR_CONFIG: r_data_reg <= config_reg;
                    ADDR_RESULT: r_data_reg <= result_reg;
                    ADDR_SSI_KEY_L: r_data_reg <= ssi_key_reg[31:0];
                    ADDR_SSI_KEY_H: r_data_reg <= ssi_key_reg[63:32];
                    ADDR_SSI_ROOT: r_data_reg <= ssi_root_reg;
                    ADDR_SSI_RES: r_data_reg <= ssi_result_reg;
                    ADDR_RNK_HASH_L: r_data_reg <= ranker_hash_reg[31:0];
                    ADDR_RNK_HASH_H: r_data_reg <= ranker_hash_reg[63:32];
                    ADDR_RNK_SEG_L: r_data_reg <= ranker_seg_reg[31:0];
                    ADDR_RNK_SEG_H: r_data_reg <= ranker_seg_reg[63:32];
                    ADDR_RNK_POS_L: r_data_reg <= ranker_pos_reg[31:0];
                    ADDR_RNK_POS_H: r_data_reg <= ranker_pos_reg[63:32];
                    ADDR_RNK_SCORE: r_data_reg <= ranker_score_reg;
                    ADDR_RNK_RES: r_data_reg <= ranker_result_reg;
                    default: begin
                        r_data_reg <= 32'h0;
                        r_resp_reg <= 2'b10;
                    end
                endcase
            end
            if (r_handshake) begin
                r_valid_reg <= 1'b0;
            end
            if (!r_valid_reg) begin
                ar_ready_reg <= 1'b1;
            end
        end
    end
    assign ssi_search_key = ssi_key_reg;
    assign ssi_root_addr = ssi_root_reg;
    assign ssi_start = control_reg[0];
    assign ranker_query_hash = ranker_hash_reg;
    assign ranker_segment_id = ranker_seg_reg;
    assign ranker_segment_pos = ranker_pos_reg;
    assign ranker_base_score = ranker_score_reg;
    assign ranker_valid = control_reg[1];
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            ssi_result_reg <= 32'h0;
            ranker_result_reg <= 32'h0;
            status_reg <= 32'h0;
        end else begin
            if (ssi_done) begin
                ssi_result_reg <= ssi_result_addr;
                status_reg[0] <= ssi_found;
                status_reg[15:8] <= ssi_depth;
            end
            if (ranker_done) begin
                ranker_result_reg <= ranker_final_score;
                status_reg[1] <= 1'b1;
                status_reg[31:16] <= ranker_rank;
            end
        end
    end
    assign ssi_node_valid = client_grant[0] && mem_ready;
    SSISearch_topEntity ssi_search (
        .clk(clk),
        .rst(reset),
        .enable(1'b1),
        .searchRequest_key(ssi_search_key),
        .searchRequest_root(ssi_root_addr),
        .searchRequest_valid(ssi_start),
        .nodeData(mem_rdata),
        .nodeValid(ssi_node_valid),
        .memAddr(ssi_mem_addr),
        .memReq(ssi_mem_req),
        .memGrant(client_grant[0]),
        .resultAddr(ssi_result_addr),
        .resultFound(ssi_found),
        .resultDepth(ssi_depth),
        .resultValid(ssi_done)
    );
    RankerCore_topEntity ranker (
        .clk(clk),
        .rst(reset),
        .enable(1'b1),
        .queryHash(ranker_query_hash),
        .segmentID(ranker_segment_id),
        .segmentPos(ranker_segment_pos),
        .baseScore(ranker_base_score),
        .inputValid(ranker_valid),
        .finalScore(ranker_final_score),
        .rank(ranker_rank),
        .outputValid(ranker_done)
    );
    assign client_req[0] = ssi_mem_req;
    assign client_req[1] = 1'b0;
    assign client_req[2] = 1'b0;
    assign client_req[3] = 1'b0;
    MemoryArbiter_topEntity mem_arbiter (
        .clk(clk),
        .rst(reset),
        .enable(1'b1),
        .client0_req(client_req[0]),
        .client0_addr(ssi_mem_addr),
        .client0_wdata(16'h0),
        .client0_we(1'b0),
        .client1_req(client_req[1]),
        .client1_addr(32'h0),
        .client1_wdata(16'h0),
        .client1_we(1'b0),
        .client2_req(client_req[2]),
        .client2_addr(32'h0),
        .client2_wdata(16'h0),
        .client2_we(1'b0),
        .client3_req(client_req[3]),
        .client3_addr(32'h0),
        .client3_wdata(16'h0),
        .client3_we(1'b0),
        .client0_grant(client_grant[0]),
        .client1_grant(client_grant[1]),
        .client2_grant(client_grant[2]),
        .client3_grant(client_grant[3]),
        .memAddr(arbiter_mem_addr),
        .memWData(arbiter_mem_wdata),
        .memWE(arbiter_mem_we),
        .memReq(arbiter_mem_req),
        .memReady(mem_ready)
    );
    assign mem_addr = arbiter_mem_addr;
    assign mem_wdata = arbiter_mem_wdata;
    assign mem_we = arbiter_mem_we;
    assign mem_oe = !arbiter_mem_we && arbiter_mem_req;
    assign mem_ce = arbiter_mem_req;
    assign led_status = { ssi_done, ranker_done, arbiter_mem_req, mem_ready, client_grant[3:0] };
    assign led_error = (status_reg[0] == 1'b0) && ssi_done;
    assign irq_out = ssi_done || ranker_done;
endmodule